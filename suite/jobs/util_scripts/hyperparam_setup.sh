#!/usr/bin/env bash

if [[ "$model_name" =~ ^t5-base|t5-large|codet5-base|codet5-large|codet5p-220m$ ]]; then
	echo "Hyperparam setup: loading specific for $model_name."
	export learning_rate="3e-3"
	export weight_decay="1e-4"
	export warmup_steps=100
	export per_device_train_batch_size=64
	export per_device_eval_batch_size=128
	export generation_batch_size=128
	export humaneval_batch_size=1
	export max_source_length=256
	export max_target_length=200
	export max_new_tokens=256
	export quantization_mode="4bit"
	export label_names="labels"
	export overwrite_output_dir=1
	export remove_unused_columns=0
	export num_beams=1
	export metric_for_best_model="loss"
	export patience=10
	export use_fast_tokenizer=1
	export eval_steps="0.1"
	export logging_steps="0.1"

elif [[ "$model_name" =~ ^llama2-7b-hf|codellama-7b$ ]]; then
	echo "Hyperparam setup: loading specific for $model_name."
	export learning_rate="2e-4"
	export weight_decay="1e-4"
	export warmup_steps=500
	export per_device_train_batch_size=16
	export per_device_eval_batch_size=32
	export generation_batch_size=32
	export humaneval_batch_size=1
	export max_source_length=256
	export max_target_length=200
	export max_new_tokens=256
	export quantization_mode="4bit"
	export label_names="labels"
	export overwrite_output_dir=1
	export remove_unused_columns=0
	export num_beams=1
	export metric_for_best_model="loss"
	export patience=10
	export use_fast_tokenizer=1
	export eval_steps="0.1"
	export logging_steps="0.1"
else
	echo "WARNING: Unknown model $model_name."
	echo "Hyperparam setup: no default for $model_name."
	# export learning_rate="5e-5"
	# export weight_decay="0.01"
	# export warmup_steps=500
	# export per_device_train_batch_size=16
	# export per_device_eval_batch_size=16
	# export max_source_length=512
	# export max_target_length=256
	# export generation_max_length=$max_target_length
fi
