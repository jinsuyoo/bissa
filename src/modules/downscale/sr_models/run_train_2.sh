CUDA_VISIBLE_DEVICES=2 python train.py --cuda --gpus 2 --model edsr --nEpochs 30 --lr 1e-4 --step 10 --lr-decay 0.5 --checkpoint "checkpoints/edsr" | tee result2.md
#CUDA_VISIBLE_DEVICES=2 python train.py --cuda --gpus 2 --model rcan --nEpochs 30 --lr 1e-4 --step 10 --lr-decay 0.5 --checkpoint "checkpoints/rcan" | tee result2.md
