#CUDA_VISIBLE_DEVICES=3 python train.py --cuda --gpus 3 --model edsr --nEpochs 30 --lr 1e-4 --step 10 --lr-decay 0.5 --checkpoint "checkpoints/edsr" | tee result3.md
CUDA_VISIBLE_DEVICES=3 python train.py --cuda --gpus 3 --model rcan --nEpochs 30 --lr 1e-4 --step 10 --lr-decay 0.5 --checkpoint "checkpoints/rcan" | tee result3.md
