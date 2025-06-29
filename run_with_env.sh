#!/bin/bash
export LD_LIBRARY_PATH=/mnt/second/qinhaoping/anaconda3/envs/kuiper/lib:$LD_LIBRARY_PATH
exec "$@" 