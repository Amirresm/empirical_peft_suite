#!/usr/bin/env bash
source "../model_vars.sh"
export dataset_name="csn-python"
export dataset_path="$storage_root/data/CodeSearchNet/python"
export data_parent_path="$model_parent_path/$dataset_name"

source "$prog_root/jobs/util_scripts/dataset_setup.sh"
