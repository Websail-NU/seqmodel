#!/bin/bash

mkdir -p _cached
echo "[0/3] Checking data..."
if ! [ -d "_cached/common_wordnet_defs" ]; then
  echo "[1/3] Downloading data..."
  wget http://websail-fe.cs.northwestern.edu/downloads/dictdef/common_wordnet_defs.tar.gz
  tar -xf common_wordnet_defs.tar.gz
  rm common_wordnet_defs.tar.gz
  mv common_wordnet_defs _cached/common_wordnet_defs
else
  echo "[1/3] Cached files found"
fi

echo "[2/3] Copying files..."
DIR="../../data/common_wordnet_defs"
DATA_NAMES="first_senses lemma_senses all_senses"
for N in $DATA_NAMES; do
  mkdir -p $DIR"/"$N
  cp "_cached/common_wordnet_defs/function_words.txt" $DIR"/"$N"/"
done
SPLITS="train valid test"
for SPLIT in $SPLITS; do
  cp "_cached/common_wordnet_defs/"$SPLIT".txt" $DIR"/all_senses/"$SPLIT".txt"
  grep -P '\t1\t' $DIR"/all_senses/"$SPLIT".txt" > $DIR"/first_senses/"$SPLIT".txt"
  grep -P '\tlemma\t' $DIR"/all_senses/"$SPLIT".txt" > $DIR"/lemma_senses/"$SPLIT".txt"
done

echo "[3/3] Preprocessing..."
for N in $DATA_NAMES; do
  python gen_vocab.py $DIR"/"$N --parallel_text --end_seq --end_encode --start_seq
  python gen_vocab.py $DIR"/"$N --parallel_text --part_indices 0 --char_level --vocab_filename char_vocab.txt
done

