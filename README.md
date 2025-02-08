# MVReward
Official Implementation of [AAAI 2025] MVReward: Better Aligning and Evaluating Multi-View Diffusion Models with Human Preferences
![image](https://github.com/victor-thu/MVReward/blob/main/assets/pipeline.png)
## Todo List
- [x] Paper Link 
- [x] Quick Start with MVReward Evaluation
- [ ] Dataset
- [ ] MVReward Checkpoint
- [ ] Inference
- [ ] Training
![image](https://github.com/victor-thu/MVReward/blob/main/assets/quantitative.png)
---

## 🚀 Quick Start with MVReward Evaluation

### 1. Clone Repository
```bash
git clone https://github.com/victor-thu/MVReward.git
cd MVReward
```
### 2. Download Model Weights

Download pretrained weights from Google Drive:  
[https://drive.google.com/file/d/1dJPnwxN9vdp2eS2aDPTSHvKunT_mibEE/view?usp=drive_link](https://drive.google.com/file/d/1dJPnwxN9vdp2eS2aDPTSHvKunT_mibEE/view?usp=drive_link)

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔧 Inference Script Usage

Run the inference script:
```bash
python get_reward.py \
    -c "path/to/checkpoint.pt" \
    -i "path/to/input_dir" \
    -s  \            # Whether to save output (True/False)
    -o "path/to/output_dir/reward_list.json"  # Output path (required if -s=True)
```

---

## ⚙️ Arguments

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--checkpoint` | `-c` | ✅ | - | Path to pretrained MVReward checkpoint (`.pt` or `.pth`) |
| `--input_dir` | `-i` | ✅ | - | Path to input data directory |
| `--save_output` | `-s` | ❌ | `False` | Whether to save outputs (`True`/`False`) |
| `--output_dir` | `-o` | Conditional | - | Output directory path (required when `-s=True`) |

---