#!/bin/bash

DATA_DIR="data/tinyshakespeare"
MAIN_DIR="curexp"
PREFIX="ts"
EXP_DIR="bw"
CONFIG_DIR="config"
BACKUP_DIR="backup"
# -------------------------------------------------------
NO_BACKUP=false
while getopts ":d" opt; do
    case "$opt" in
        d)
            NO_BACKUP= true
            ;;
    esac
done

EXP_DIR="$PREFIX-$EXP_DIR"
CONFIG_DIR="$PREFIX-$CONFIG_DIR"
BACKUP_DIR="$PREFIX-$BACKUP_DIR"

mkdir -p "$MAIN_DIR/$BACKUP_DIR"

if [ -d "$MAIN_DIR/$EXP_DIR" ]; then
    if $NO_BACKUP ; then
        rm -r "$MAIN_DIR/$EXP_DIR"
    else
        mv "$MAIN_DIR/$EXP_DIR" "$MAIN_DIR/$BACKUP_DIR/$EXP_DIR-$(date +%y%m%dT%H%M%S)"
    fi
fi
# -------------------------------------------------------
python main_lm.py train "$DATA_DIR" "$MAIN_DIR/$EXP_DIR" \
--log_level debug \
--load_model_opt "$MAIN_DIR/$CONFIG_DIR/model_opt_gru.json" \
--load_train_opt "$MAIN_DIR/$CONFIG_DIR/train_opt_adam.json" \
--batch_size 64 \
--seq_len 35 \
--char_data \
--load_checkpoint "$MAIN_DIR/ts-v/checkpoint/best" \
--relax_ckp_restore \
--rnn:use_bw_state \
--train:max_epoch 10
