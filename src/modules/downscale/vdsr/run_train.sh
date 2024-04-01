# Default - MSE, SGD, gradient clip
CUDA_VISIBLE_DEVICES=3 python custom_train.py --cuda --gpus 3 --nEpochs 50 --lr 0.1 --step 10 --lr-decay 0.1 --checkpoint "checkpoint/base" | tee result_base.md

# OURS
CUDA_VISIBLE_DEVICES=3 python custom_train.py --cuda --gpus 3 --nEpochs 150 --lr 1e-4 --step 50 --lr-decay 0.5 --checkpoint "checkpoint/150_50" | tee result_150_50.md

#CUDA_VISIBLE_DEVICES=3 python custom_train.py --cuda --gpus 3 --nEpochs 1000 --lr 1e-4 --step 100 --lr-decay 0.5 --checkpoint "checkpoint/1000_100" | tee result_1000_100.md
