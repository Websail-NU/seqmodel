#!/bin/bash

CORPUS=$1
VERSION=$2

PRE_PATH="explm/stat/$CORPUS-pre"
TARGET_PATH="explm/stat/$CORPUS-$VERSION"

if [ -d $TARGET_PATH ]; then
    rm -r $TARGET_PATH
fi
cp -r $PRE_PATH $TARGET_PATH
mkdir $TARGET_PATH"/decode"
