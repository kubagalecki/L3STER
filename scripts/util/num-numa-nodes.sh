#!/bin/bash

lscpu | grep -i numa | head -1 | grep -o "[0-9]*" | tr -d '\n\r'
