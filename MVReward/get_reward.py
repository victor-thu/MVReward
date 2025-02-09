import os
import sys
import time
import json
import argparse
from tqdm import tqdm
# Add the parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import sys

# sys.stderr = open('error.log', 'w')

from PIL import Image
from models.mvreward.mvreward import mvreward
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# For reproducibility
set_seed(42)

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def parse_args():
    parser = argparse.ArgumentParser(description='run reward model')

    parser.add_argument('-c', '--checkpoint',
                        type=str, 
                        default='checkpoints/mvreward_b.pt')

    parser.add_argument('-i', '--input_dir',
                        type=str,
                        default='imgs')
    
    parser.add_argument('-s', '--save_output',
                        type=bool,
                        default=False)
    
    parser.add_argument('-o', '--output_dir',
                        type=str,
                        default='outputs')
    
    return parser.parse_args()
    
def load_reward_model(model:torch.nn.Module, checkpoint_path:str, is_DDP_weight=True):
    print('load checkpoint from %s'%checkpoint_path)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if is_DDP_weight:
        # Remove 'module' prefix from checkpoint keys
        warped_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('module.', '')  # remove 'module.' prefix
            warped_state_dict[new_key] = v
        msg = model.load_state_dict(warped_state_dict, strict=False)
    else:
        msg = model.load_state_dict(state_dict, strict=False)
    print(msg)

    return model

@torch.no_grad()
def get_methods_rewards(model, device, input_dir, save_output, output_dir):
    model.eval()
    # train_path = 'data/imgs'
    prompts = os.listdir(input_dir)
    bar = tqdm(range(len(prompts)), desc=f'Calculating methods rewards: ')
    reward_list = []
    methods = ['envision3d', 'era3d', 'zeroplus', 'wonder3d', 'objaverse']
    preprocess = _transform(224)
    for prompt in prompts:
        # get input image tensor
        input_path = os.path.join(input_dir, prompt, 'input.png')
        input_img = Image.open(input_path)
        input_tensor = preprocess(input_img).unsqueeze(0).to(device)

        # get train images tensor
        for method in os.listdir(os.path.join(input_dir, prompt)):
            if method in methods:
                train_img_tensors = []
                train_prompt_path = os.path.join(input_dir, prompt, method)
                img_names = os.listdir(train_prompt_path)
                sorted_img_names = sorted(img_names, key=lambda x: int(''.join(filter(str.isdigit, x))))
                for train_img in sorted_img_names:
                    train_img_path = os.path.join(train_prompt_path, train_img)
                    train_img = Image.open(train_img_path)
                    train_img_tensor = preprocess(train_img).unsqueeze(0)
                    train_img_tensors.append(train_img_tensor)
                train_imgs_tensor = torch.cat(train_img_tensors, dim=0).to(device)
        
                reward_train = model.rate_mvimgs(input_tensor, train_imgs_tensor).detach().cpu().numpy().item()
        
                reward_list.append({'prompt': prompt, 'method':method, 'reward': reward_train})

        if save_output:
            with open(output_dir, 'w') as f:
                json.dump(reward_list, f, indent=4)
                f.write("\n")
        bar.update(1)

        return reward_list


if __name__ == '__main__':
    args = parse_args()
    
    if not os.path.isfile(args.checkpoint):
        raise ValueError(f"Invalid checkpoint file path: {args.checkpoint}")
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Invalid input directory path: {args.input_dir}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = mvreward(device=device).to(device)
    model = load_reward_model(model, args.checkpoint)
    get_methods_rewards(model, device, args.input_dir, args.save_output, args.output_dir)


