#!/bin/bash


python -m flexgen.flex_opt \
--model facebook/opt-175b \
--path _DUMMY_ --prompt-len 256 \
--gen-len 32 \
--pin-weight 0 \
--percent 0 50 0 0 0 100 \
--gpu-batch-size 64 \
--num-gpu-batches 8 \
--cpu \
--debug fewer_batch

# python -m flexgen.flex_opt --model facebook/opt-175b --path _DUMMY_ --prompt-len 256 --gen-len 32 --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 64 --num-gpu-batches 8 --cpu --debug fewer_batch

python -m flexgen.flex_opt \
--model facebook/opt-6.7b \
--path _DUMMY_ --prompt-len 256 \
--gen-len 32 \
--percent 100 0 100 0 100 0 \
--gpu-batch-size 4 \
--overlap False

# python -m flexgen.flex_opt --model facebook/opt-6.7b --path _DUMMY_ --prompt-len 256 --gen-len 32 --percent 100 0 100 0 100 0 --gpu-batch-size 4 --overlap False

