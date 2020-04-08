# Copyright (c) 2019 NVIDIA Corporation
#!/usr/bin/env bash
if [ $# -eq 1 ]; then
    EXAMPLE_DATA=$1
    echo "Building LM inside $EXAMPLE_DATA"
else
    echo "Example data directory needed."
fi
cd $EXAMPLE_DATA

# create folder
set -e
if [ ! -d "language_model" ]; then
  mkdir language_model
fi

# download librispeech dataset
cd language_model
if [ ! -f "librispeech-lm-norm.txt" ]; then
    wget http://www.openslr.org/resources/11/librispeech-lm-norm.txt.gz
    gzip -d librispeech-lm-norm.txt.gz
fi

# convert all upper case characters to lower case
echo "Text formatting..."
tr '[:upper:]' '[:lower:]' < librispeech-lm-norm.txt > 6-gram.txt

cd /tmp/NeMo/scripts/
# build a language model
python build_lm_text.py $EXAMPLE_DATA/language_model/6-gram.txt --n 6

# delete arpa file
cd $EXAMPLE_DATA/language_model/
rm 6-gram.arpa