#!/bin/bash

if [ -n "$SLURM_CPUS_ON_NODE" ]; then
  echo "$SLURM_CPUS_ON_NODE" | tr -d '\n\r'
else
  grep ^cpu\\scores /proc/cpuinfo | uniq | grep -o "[0-9]*" | tr -d '\n\r'
fi
