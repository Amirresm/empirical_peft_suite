#!/usr/bin/env bash

export output_path="$data_parent_path/$config_title"
export logging_path="$output_path/logs"
export adapter_config=$peft_name
export generation_output_path="$output_path/gen_output"
export generation_max_length=$max_target_length

evaluation_strategy="steps"
source "../../../control_vars.sh"
if [[ "$do_eval" == 0 ]]; then
	evaluation_strategy="no"
fi

if [[ "$lib_name" == "ah" ]]; then
	echo "Using AdapterHub..."
	export use_adapterhub=1
	export CUDA_VISIBLE_DEVICES=0
fi

mkdir -p "$output_path"
mkdir -p "$logging_path"
touch "${output_path}/memuse.txt"
touch "$output_path/job_report.log"

printenv >"${output_path}/env_vars.txt"
pip freeze >"$output_path/requirements.txt"
pip freeze -l >"$output_path/requirements-local.txt"
python --version >"$output_path/python_version.txt"

"${prog_root}/checkmem.sh" $memcheck_interval "${output_path}/memuse.txt" &

echo "Bash going to Python..."
python3 "$script_path" \
	--config_title "$config_title" \
	--model_name_or_path "$model_path" \
	--tokenizer_name_or_path "$tokenizer_name_or_path" \
	--do_train "$do_train" \
	--do_eval "$do_eval" \
	--do_predict "$do_predict" \
	--train_file "${train_file}" \
	--validation_file "${eval_file}" \
	--test_file "${test_file}" \
	--additional_predict_dataset_paths "$additional_predict_dataset_paths" \
	--text_column "$text_column" \
	--summary_column "$summary_column" \
	--text_tokenized "$text_tokenized" \
	--summary_tokenized "$summary_tokenized" \
	--source_prefix "$source_prefix" \
	--output_dir "$output_path" \
	--overwrite_output_dir "$overwrite_output_dir" \
	--use_fast_tokenizer "$use_fast_tokenizer" \
	--train_tokenizer "$train_tokenizer" \
	--per_device_train_batch_size "$per_device_train_batch_size" \
	--per_device_eval_batch_size "$per_device_eval_batch_size" \
	--learning_rate "$learning_rate" \
	--weight_decay "$weight_decay" \
	--num_train_epochs "$num_train_epochs" \
	--warmup_steps "$warmup_steps" \
	--predict_with_generate \
	--evaluation_strategy $evaluation_strategy \
	--eval_steps "$eval_steps" \
	--humaneval_num "$humaneval_num" \
	--logging_strategy steps \
	--logging_steps "$logging_steps" \
	--logging_dir "$logging_path" \
	--report_to "$report_to" \
	--save_total_limit "$save_total_limit" \
	--remove_unused_columns "$remove_unused_columns" \
	--num_beams "$num_beams" \
	--max_new_tokens "$max_new_tokens" \
	--metric_for_best_model "$metric_for_best_model" \
	--label_names "$label_names" \
	--patience "$patience" \
	--load_best_model_at_end "$load_best_model_at_end" \
	--metric_path "$bleu_path" \
	--metric_path_alt "$rouge_path" \
	--quantization_mode "$quantization_mode" \
	--train_adapter "$train_adapter" \
	--adapter_config "$adapter_config" \
	--adapter_path "$adapter_path" \
	--preload_adapter "$preload_adapter" \
	--generation_output_path "$generation_output_path" \
	--max_source_length "$max_source_length" \
	--max_target_length "$max_target_length" \
	--generation_max_length "$generation_max_length" \
	--pad_to_max_length "$pad_to_max_length" \
	--ignore_pad_token_for_loss "$ignore_pad_token_for_loss" \
	--max_train_samples "$max_train_samples" \
	--max_eval_samples "$max_eval_samples" \
	--max_predict_samples "$max_predict_samples" \
	--use_adapterhub "$use_adapterhub" \
	--advfusion_paths "$advfusion_paths" \
	--advfusion_target "$advfusion_target" \
	--generation_batch_size "$generation_batch_size" \
	--humaneval_batch_size "$humaneval_batch_size" \
	--humaneval_prompt_mode "$humaneval_prompt_mode" \
	--preprocessing_num_workers "$preprocessing_num_workers" \
	2>&1 | tee "$output_path/job_report.log"
