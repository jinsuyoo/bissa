#CUDA_VISIBLE_DEVICES=2 python custom_eval_set5.py --cuda --gpus 2 --model model/set5/official_pretrained.pth
#CUDA_VISIBLE_DEVICES=2 python custom_eval_set5.py --cuda --gpus 2 --model model/set5/official_code.pth
CUDA_VISIBLE_DEVICES=2 python custom_eval_set5.py --cuda --gpus 2 --model checkpoint/model_epoch_50.pth
