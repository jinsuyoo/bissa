# BIx4 to GT
#CUDA_VISIBLE_DEVICES=3 python custom_eval.py --cuda --gpus 3 --model model/official_pretrained.pth --dataset BIx4

# P_GT to GT
#CUDA_VISIBLE_DEVICES=3 python custom_eval.py --cuda --gpus 3 --model checkpoint/model_epoch_40.pth
#CUDA_VISIBLE_DEVICES=3 python custom_eval.py --cuda --gpus 3 --model checkpoint/model_epoch_150.pth
#CUDA_VISIBLE_DEVICES=2 python custom_eval.py --cuda --gpus 2 --model checkpoint/model_epoch_50.pth
CUDA_VISIBLE_DEVICES=2 python custom_eval.py --cuda --gpus 2 --model checkpoint/1000_100/model_epoch_500.pth
