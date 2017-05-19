#!/bin/bash
mkdir -p _cached
echo "[0/3] Checking data..."
if ! [ -d "_cached/simple-examples" ]; then
  echo "[1/3] Downloading data..."
  wget http://websail-fe.cs.northwestern.edu/downloads/cached/simple-examples.tgz
  tar -xf simple-examples.tgz
  rm simple-examples.tgz
  mv simple-examples _cached/simple-examples
else
  echo "[1/3] Cached files found"
fi
echo "[2/3] Copying files..."
mkdir -p ../../data/ptb
cp _cached/simple-examples/data/ptb.test.txt ../../data/ptb/test.txt
cp _cached/simple-examples/data/ptb.train.txt ../../data/ptb/train.txt
cp _cached/simple-examples/data/ptb.valid.txt ../../data/ptb/valid.txt
mkdir -p ../../data/ptb-char
cp _cached/simple-examples/data/ptb.char.test.txt ../../data/ptb-char/test.txt
cp _cached/simple-examples/data/ptb.char.train.txt ../../data/ptb-char/train.txt
cp _cached/simple-examples/data/ptb.char.valid.txt ../../data/ptb-char/valid.txt
echo "[3/3] Generating vocab files..."
python gen_vocab.py ../../data/ptb/ --end_seq
python gen_vocab.py ../../data/ptb-char/ --end_seq
