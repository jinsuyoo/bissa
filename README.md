# Demo code

"Looking Beyond Input Frames: Self-Supervised Adaptation for Video Super-Resolution"

## Setup

1. Install dependencies using:
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install opencv-python tqdm pyyaml
```

2. Download pre-trained EDVR parameter (i.e., EDVR_L_x4_SR_Vimeo90K_official-162b54e4.pth) from the official EDVR repository and place it under "pre-trained" directory.

## Usage

Run following command:
```bash
python adapt_single_video.py -opt options/EDVR/adapt_vid4_calendar.yml
```

Results will be saved inside the 'experiments' directory.

## Acknowledgement

The codes are based on: 
- [EDVR](https://github.com/xinntao/EDVR)

Thanks for open sourcing such a wonderful work!