#!/usr/bin/env bash
source "../../job_vars.sh"
export model_name="codellama-7b"
export model_path="$storage_root/models/$model_name"
export model_parent_path="$job_path_path/$model_name"

source "$prog_root/jobs/util_scripts/hyperparam_setup.sh"
