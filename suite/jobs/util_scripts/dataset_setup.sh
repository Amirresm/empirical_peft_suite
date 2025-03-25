#!/usr/bin/env bash

echo "Processing dataset: $dataset_name"

if [[ "$dataset_name" == "conpy" ]]; then
	export source_prefix=""
	export text_column="rewritten_intent"
	export summary_column="snippet"
	export text_tokenized=0
	export summary_tokenized=0
	export num_train_epochs="4.0"

	export do_train=1
	export do_eval=0
	export do_predict=1

	export train_file="${dataset_path}/conala-train.json"
	export eval_file="${dataset_path}/conala-test.json"
	export test_file="${dataset_path}/conala-test.json"
elif [[ "$dataset_name" == "spp_450k" ]]; then
	export source_prefix=""
	export text_column="prompt"
	export summary_column="NONE"
	export text_tokenized=0
	export summary_tokenized=0
	export num_train_epochs="5.0"

	export do_train=1
	export do_eval=0
	export do_predict=1

	export train_file="${dataset_path}/SPP_450k_unverified.jsonl"
	export eval_file="SPLIT0.001"
	export test_file="SPLIT0.10"
elif [[ "$dataset_name" == "spp_30k" ]]; then
	export source_prefix=""
	export text_column="code"
	export summary_column="NONE"
	export text_tokenized=0
	export summary_tokenized=0
	export num_train_epochs="5.0"

	export do_train=1
	export do_eval=0
	export do_predict=1

	export train_file="${dataset_path}/SPP_30k_verified.jsonl"
	export eval_file="SPLIT0.01"
	export test_file="SPLIT0.05"
elif [[ "$dataset_name" == "sppu_30k" ]]; then
	export source_prefix=""
	export text_column="code"
	export summary_column="NONE"
	export text_tokenized=0
	export summary_tokenized=0
	export num_train_epochs="5.0"

	export do_train=1
	export do_eval=0
	export do_predict=1

	export train_file="${dataset_path}/SPP_450k_unverified_sample.jsonl"
	export eval_file="SPLIT0.01"
	export test_file="SPLIT0.05"
elif [[ "$dataset_name" == "csn" ]]; then
	export source_prefix=""
	export text_column="func_code_tokens"
	export summary_column="func_documentation_tokens"
	export text_tokenized=1
	export summary_tokenized=1
	export num_train_epochs="1.0"

	export do_train=1
	export do_eval=1
	export do_predict=1

	export train_file="${dataset_path}/train.jsonl"
	export eval_file="${dataset_path}/validation.jsonl"
	export test_file="${dataset_path}/test.jsonl"
elif [[ "$dataset_name" =~ ^csn-python|csn-php|csn-javascript|csn-java|csn-go|csn-ruby$ ]]; then
	export source_prefix=""
	export text_column="code_tokens"
	export summary_column="docstring_tokens"
	export text_tokenized=1
	export summary_tokenized=1
	export num_train_epochs="1.0"

	export do_train=1
	export do_eval=1
	export do_predict=1

	export train_file="${dataset_path}/train.jsonl"
	export eval_file="${dataset_path}/valid.jsonl"
	export test_file="${dataset_path}/test.jsonl"
elif [[ "$dataset_name" =~ ^rsum-combined$ ]]; then
	export source_prefix=""
	export text_column="code_tokens"
	export summary_column="docstring_tokens"
	export text_tokenized=1
	export summary_tokenized=1
	export num_train_epochs="5.0"

	export do_train=1
	export do_eval=1
	export do_predict=1

	export train_file="${dataset_path}/train.jsonl"
	export eval_file="${dataset_path}/valid.jsonl"
	export test_file="${dataset_path}/test.jsonl"
elif [[ "$dataset_name" =~ ^multiplt-r$ ]]; then
	export source_prefix=""
	export text_column="prompt"
	export summary_column="completion"
	export text_tokenized=0
	export summary_tokenized=0
	export num_train_epochs="5.0"

	export do_train=1
	export do_eval=1
	export do_predict=1

	export train_file="${dataset_path}/multiplt-r.jsonl"
	export eval_file="SPLIT0.01"
	export test_file="SPLIT0.05"
else
	echo "Dataset setup failed: unknown dataset name $dataset_name. Exiting..."
fi
