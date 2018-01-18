#!/bin/bash

DATA_DIR="data/tinyshakespeare"
MAIN_DIR="curexp"
PREFIX="ts"
EXP_DIR="bw-m2-gmm"
CONFIG_DIR="config"
BACKUP_DIR="backup"
# -------------------------------------------------------
# <NO EDIT>
NO_BACKUP=false
EVAL_MODE=false
COMMAND="train"
EXP_DIR="$PREFIX-$EXP_DIR"
CONFIG_DIR="$PREFIX-$CONFIG_DIR"
BACKUP_DIR="$PREFIX-$BACKUP_DIR"
while getopts "de" opt; do
    case "$opt" in
        d)
            NO_BACKUP=" "true
            ;;
        e)
            EVAL_MODE=" "true
            ;;
    esac
done
mkdir -p "$MAIN_DIR/$BACKUP_DIR"
if $EVAL_MODE ; then
    COMMAND="eval"
else
    COMMAND="train"
    if [ -d "$MAIN_DIR/$EXP_DIR" ]; then
        if $NO_BACKUP ; then
            rm -r "$MAIN_DIR/$EXP_DIR"
        else
            mv "$MAIN_DIR/$EXP_DIR" "$MAIN_DIR/$BACKUP_DIR/$EXP_DIR-$(date +%y%m%dT%H%M%S)"
        fi
    fi
fi
# </NO_EDIT>
# -------------------------------------------------------
python main_lm.py "$COMMAND" "$DATA_DIR" "$MAIN_DIR/$EXP_DIR" \
--gpu \
--log_level debug \
--load_model_opt "$MAIN_DIR/$CONFIG_DIR/model_opt_gru.json" \
--load_train_opt "$MAIN_DIR/$CONFIG_DIR/train_opt_adam.json" \
--batch_size 65 \
--seq_len 35 \
--char_data \
--relax_ckp_restore \
--train:max_epoch 20 \
--train:init_lr 0.001 \
--lr:decay_factor 1.0 \
--load_checkpoint "$MAIN_DIR/ts-v/checkpoint/best" \
--xxx:full_seq_lookup \
--xxx:add_first_token \
--rnn:use_bw_state \
--rnn:q_mode "gmm"  \
--train:grad_vars_contain "/bw/" \
--random_seq_len \
--random_seq_len_min 5 \
--random_seq_len_max 20 \
--rnn:gmm_path "curexp/ts-v/gmm64valid_model.pkl"
# --eval_file "valid.txt"
