#!/bin/bash

# http://websail-fe.cs.northwestern.edu/downloads/dictdef/wordnet_lemma_cv.tar.gz

mkdir -p _cached
echo "[0/3] Checking data..."
if ! [ -d "_cached/wordnet_lemma_cv" ]; then
  echo "[1/3] Downloading data..."
  wget http://websail-fe.cs.northwestern.edu/downloads/dictdef/wordnet_lemma_cv.tar.gz
  tar -xf wordnet_lemma_cv.tar.gz
  rm wordnet_lemma_cv.tar.gz
  mv wordnet_lemma_cv _cached/wordnet_lemma_cv
else
  echo "[1/3] Cached files found"
fi

echo "[2/3] Copying files..."
DIR="../../data/wordnet_lemma_cv"
cp -r _cached/wordnet_lemma_cv '../../data'
# rm -r $DIR"/wordnet_lemma_cv"

echo "[3/3] Preprocessing..."
python gen_vocab.py $DIR"/cv/cv0/" --parallel_text --end_seq --end_encode --start_seq --text_filenames "train.txt,valid.txt"
python gen_vocab.py $DIR"/cv/cv0/" --parallel_text --part_indices 0 --char_level --vocab_filename char_vocab.txt --text_filenames "train.txt,valid.txt"

for f in "$DIR/cv/cv"*; do
    if [[ $f == *"cv0"* ]]; then
        continue
    fi
    cp $DIR"/cv/cv0/"*_vocab.txt $f"/"
done

cp $DIR"/cv/cv0/"*_vocab.txt $DIR"/cv"

mv "$DIR/cv/cv"* "$DIR/"
mv "$DIR/cv" "$DIR/splits"
