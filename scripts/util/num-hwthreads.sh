#!/bin/bash

if [ -n "$SLURM_CPUS_ON_NODE" ]; then
  threads_per_core=$(lscpu | grep "Thread(s) per core:" | grep -hoE "\w$" | tr -d '\n\r')
  echo $((SLURM_CPUS_ON_NODE * threads_per_core)) | tr -d '\n\r'
else
  grep -c ^processor /proc/cpuinfo | uniq | grep -o "[0-9]*" | tr -d '\n\r'
fi
