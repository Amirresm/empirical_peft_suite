#!/usr/bin/env bash

# use concurrency to run the test
# run the test in parallel

concurrency=6

parallel -j "$concurrency" python stat_test.py '{= uq =}' :::: ./run_stat_args.txt
