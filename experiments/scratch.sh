#!/bin/bash

EXP_DIR='tmp11'
rm -r $EXP_DIR
# python main_lm.py train data/ptb $EXP_DIR \
# --gpu --log_level debug --batch_size 64 --seq_len 35 \
# --emb:dim 200 --cell:num_units 200 --cell:num_layers 2 \
# --logit:add_project  --logit:project_keep_prob 0.75 \
# --cell:cell_class tensorflow.contrib.rnn.GRUBlockCellV2 \
# --cell:in_keep_prob 0.75 --cell:out_keep_prob 0.75 \
# --lr:decay_every 1 --lr:decay_factor 1.0 --lr:start_decay_at 1 \
# --train:max_epoch 20 --train:init_lr 0.003 --reset_state_prob 0.0 \
# --model_class seqmodel.VAESeqModel
# --logit:project_act tensorflow.nn.tanh
# --share:input_emb_logit

python main_lm.py train data/ptb $EXP_DIR \
--gpu --log_level debug --batch_size 20 --seq_len 35 \
--emb:dim 200 --cell:num_units 200 --cell:num_layers 2 \
--cell:cell_class tensorflow.contrib.rnn.GRUBlockCellV2 \
--share:input_emb_logit \
--load_train_opt "curexp/ptb-config/train_opt.json" \
--cell:reset_state_prob 0.10 --cell:init_state_trainable \
--model_class seqmodel.UnigramSeqModelH
#--reset_state_prob 0.1
