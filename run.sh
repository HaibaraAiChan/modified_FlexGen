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
python -m flexgen.flex_opt \
--model facebook/opt-6.7b-muhd-64 \
--path _DUMMY_ --prompt-len 256 \
--gen-len 32 \
--percent 100 0 100 0 100 0 \
--gpu-batch-size 4 \
--overlap False

python -m flexgen.flex_opt --model facebook/opt-6.7b-muhd-64 --path _DUMMY_ --prompt-len 256 --gen-len 32 --percent 100 0 100 0 100 0 --gpu-batch-size 4 --overlap False


python -m flexgen.flex_opt --model facebook/opt-6.7b --path _DUMMY_ --prompt-len 256 --gen-len 32 --percent 100 0 100 0 100 0 --gpu-batch-size 4 --overlap False

# int4 compression
python -m flexgen.flex_opt --model facebook/opt-6.7b --path _DUMMY_ --prompt-len 256 --gen-len 32 --percent 100 0 100 0 100 0 --gpu-batch-size 4 --overlap False --compress-weight --compress-cache
python -m flexgen.flex_opt --model facebook/opt-6.7b --path _DUMMY_ --prompt-len 256 --gen-len 32 --percent 100 0 100 0 100 0 --gpu-batch-size 128 --overlap False --compress-weight --compress-cache

python -m flexgen.flex_opt --model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --gpu-batch-size 72 --overlap False --compress-weight --compress-cache

python -m flexgen.flex_opt --model facebook/opt-6.7b --path _DUMMY_ --percent 100 0 100 0 100 0 --gpu-batch-size 28 --overlap False --compress-weight --compress-cache --prompt-len 1024

python -m flexgen.flex_opt --model facebook/opt-175b --path _DUMMY_ --prompt-len 256 --gen-len 32 --pin-weight 0 --percent 0 100 0 100 0 100 --gpu-batch-size 96 --num-gpu-batches 3 --debug fewer_batch --compress-weight --compress-cache

python -m flexgen.flex_opt --model facebook/opt-175b --path _DUMMY_ --prompt-len 256 --gen-len 32 --pin-weight 0 --percent 0 50 0 0 0 100 --gpu-batch-size 64 --num-gpu-batches 8 --cpu --debug fewer_batch




#30 B model weights 
python -m flexgen.flex_opt --model facebook/opt-30b --path _DUMMY_ --prompt-len 256 --gen-len 32 --percent 10 90 0 100 0 100 --gpu-batch-size 160 --num-gpu-batches 2 --cpu --debug fewer_batch
