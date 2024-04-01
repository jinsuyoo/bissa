CUDA_VISIBLE_DEVICES=2 python custom_train_set5.py --cuda --gpus 2 --color "ycbcr" --nEpochs 50 --lr 0.1 --step 10 --lr-decay 0.1 | tee training_set5_ycbcr.md
#CUDA_VISIBLE_DEVICES=2 python custom_train_set5.py --cuda --gpus 2 | tee training_set5_rgb.md
