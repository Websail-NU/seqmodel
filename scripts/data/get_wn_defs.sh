#!/bin/bash

mkdir -p _cached
echo "[0/4] Checking data..."
if ! [ -d "_cached/common_wordnet_defs" ]; then
  echo "[1/4] Downloading data..."
  wget http://websail-fe.cs.northwestern.edu/downloads/dictdef/common_wordnet_defs.tar.gz
  tar -xf common_wordnet_defs.tar.gz
  rm common_wordnet_defs.tar.gz
  mv common_wordnet_defs _cached/common_wordnet_defs
else
  echo "[1/4] Cached files found"
fi

echo "[2/4] Copying files..."
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

echo "[3/4] Preprocessing..."
for N in $DATA_NAMES; do
  python generate_vocab.py $DIR"/"$N --parallel_text --start_seq --end_seq --end_encode --start_decode
  python extract_def_features.py $DIR"/"$N
done

echo "[4/4] Creating pruned corpus..."
for N in $DATA_NAMES; do
  mkdir -p $DIR"/"$N"_pruned"
  cp $DIR"/"$N"/function_words.txt" $DIR"/"$N"_pruned/"
  python prune_def_data.py $DIR"/"$N $DIR"/"$N"_pruned" 2
  python generate_vocab.py $DIR"/"$N"_pruned" --parallel_text --start_seq --end_seq --end_encode --start_decode
  python extract_def_features.py $DIR"/"$N"_pruned"
done
