#!/bin/bash

mkdir -p _cached
echo "[0/3] Checking data..."
if ! [ -d "_cached/wn_lemma_senses" ]; then
  echo "[1/3] Downloading data..."
  wget http://websail-fe.cs.northwestern.edu/downloads/dictdef/wn_lemma_senses.tar.gz
  tar -xf wn_lemma_senses.tar.gz
  rm wn_lemma_senses.tar.gz
  mv wn_lemma_senses _cached/wn_lemma_senses
else
  echo "[1/3] Cached files found"
fi

echo "[2/3] Copying files..."
DIR="../../data/wn_lemma_senses"
mkdir -p $DIR
SPLITS="train valid test"
for SPLIT in $SPLITS; do
  cp "_cached/wn_lemma_senses/"$SPLIT".txt" $DIR"/"$SPLIT".txt"
done

echo "[3/3] Preprocessing..."
python gen_vocab.py $DIR --parallel_text --end_seq --end_encode --start_seq
python gen_vocab.py $DIR --parallel_text --part_indices 0 --char_level --vocab_filename char_vocab.txt

