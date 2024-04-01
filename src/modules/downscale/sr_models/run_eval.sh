# P_GT to GT
CUDA_VISIBLE_DEVICES=2 python eval.py --cuda --gpus 2 --model checkpoints/edsr/model_epoch_30.pth
#CUDA_VISIBLE_DEVICES=2 python eval.py --cuda --gpus 2 --model checkpoints/rcan/model_epoch_30.pth
