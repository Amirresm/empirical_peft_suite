#!/usr/bin/env bash
source "../model_vars.sh"
export dataset_name="spp_30k"
export dataset_path="$storage_root/data/spp_30k"
export data_parent_path="$model_parent_path/$dataset_name"

source "$prog_root/jobs/util_scripts/dataset_setup.sh"
