#!/bin/bash

NGRAM_COUNT=/websail/common/tools/srilm/bin/i686-m64/ngram-count
VOCAB=$1
TEXT=$2
OUTFILE=$3
ORDER=$4

$NGRAM_COUNT -order $ORDER -text $TEXT -kndiscount -interpolate -lm $OUTFILE.arpa -vocab $VOCAB -unk -wbdiscount1 -gt1min 0 -gt2min 0 -gt3min 0 -gt4min 0
# $NGRAM_COUNT -order $ORDER -text $TEXT -interpolate -lm $OUTFILE.arpa -vocab $VOCAB -unk -wbdiscount1 -gt1min 0 -gt2min 0 -gt3min 0 -gt4min 0
$NGRAM_COUNT -order $ORDER -text $TEXT -write $OUTFILE.count -vocab $VOCAB -unk

