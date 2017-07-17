#!/bin/bash

EXP_DIR=$1
# DATA_DIR="data/common_wordnet_defs/lemma_senses"
# MODEL_OPT="../experiment/dm/$EXP_DIR/model_opt.json"
# CHECKPOINT="../experiment/dm/$EXP_DIR/checkpoint/best"
# OUT_DIR="../experiment/dm/$EXP_DIR/decode"
DATA_DIR="data/wn_lemma_senses"
MODEL_OPT="../experiment/dm2/$EXP_DIR/model_opt.json"
CHECKPOINT="../experiment/dm2/$EXP_DIR/checkpoint/best"
OUT_DIR="../experiment/dm2/$EXP_DIR/decode"


MAIN="main_word2def.py"
# MAIN="main_seq2seq.py"
LOG_DIR="tmp"
SPLITS="test valid train"
MODES="greedy sampling"

# SPLITS="test"
# MODES="greedy"

if ! [ -d $OUT_DIR ]; then
  mkdir -p $OUT_DIR
fi

for MODE in $MODES; do
    if [ "$MODE" == "greedy" ]; then
        M="--decode:greedy"
    else
        M=""
    fi
    echo $MODE
    for SPLIT in $SPLITS; do
        echo $SPLIT
        python $MAIN decode $DATA_DIR $LOG_DIR --load_model_opt $MODEL_OPT \
             --load_checkpoint $CHECKPOINT --gpu --batch_size 128 --eval_file $SPLIT.txt \
             --decode:outpath $OUT_DIR/$MODE'_'$SPLIT.txt $M
    done
done
