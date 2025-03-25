#!/usr/bin/env bash

source "./data_vars.sh"

export peft_name="ia3"
export lib_name="pl"
export remark="norm"

export config_title="${remark}_${job_name}_${model_name}_${dataset_name}_${lib_name}_${peft_name}"

export base_config_title=$config_title
export base_output_path="$data_parent_path/$base_config_title"

export adapter_path="$base_output_path"
export tokenizer_name_or_path="$base_output_path/${base_config_title}_tokenizer"

export per_device_train_batch_size=8

export do_train=1
export do_eval=1
export do_predict=1

export train_adapter=1
export preload_adapter=0

"$main_script_path"
