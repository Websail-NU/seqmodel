#!/bin/bash
mkdir -p _cached
echo "[0/3] Checking data..."
if ! [ -d "_cached/wikitext" ]; then
  echo "[1/3] Downloading data..."
  wget http://websail-fe.cs.northwestern.edu/downloads/cached/wikitext.tar.gz
  tar -xf wikitext.tar.gz
  rm wikitext.tar.gz
  mv wikitext _cached/wikitext
else
  echo "[1/3] Cached files found"
fi
echo "[2/3] Copying files..."
CORPORA="wikitext-2 wikitext-103"
for C in $CORPORA; do
    cp -r "_cached/wikitext/"$C ../../data/
done
echo "[3/3] Generating vocab files..."
for C in $CORPORA; do
    python gen_vocab.py "../../data/"$C --end_seq
done

