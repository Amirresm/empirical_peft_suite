import os
import evaluate
import torch
from adapters import (
    AdapterTrainer,
    Seq2SeqAdapterTrainer,
)
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Trainer,
    default_data_collator,
    set_seed,
)

# from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from adapter_utils import (
    init_ah_adapter,
    init_ah_advfusion,
    load_ah_adapter,
    save_ah_adapter,
)
from advf_utils import (
    unfreeze_reload_adapter,
    zero_freeze_adapter,
)
from arg_utils import parse_arguments
from constants import summarization_name_mapping
from dataset_utils import (
    get_decoder_only_preprocessor,
    get_encoder_decoder_preprocessor,
    get_generation_preprocessor,
    get_text_grouper,
    load_raw_datasets,
)
from evaluation_utils import get_compute_metrics, preprocess_logits_for_metrics
from general_utits import (
    CudaTimer,
    check_dependencies,
    check_nltk_data,
    check_version,
    handle_last_checkpoint,
)
from generation_utils import (
    generation_decoder_only,
    generation_from_predict_encoder_decoder,
    run_humaneval,
)
from init_utils import (
    ensure_decoder_only_padding_token,
    ensure_decoder_start_token,
    ensure_embedding_size,
    ensure_multilingual_tokenizer,
    get_training_corpus,
    init_config,
    init_model,
    init_tokenizer,
    train_tokenizer,
)
from logging_utils import logger, setup_logging
from peft_utils import init_and_load_peft_adapter, init_peft_adapter
from train_utils import handle_metrics

has_codebleu = False
# try:
#     from codebleu import calc_codebleu
#     has_codebleu = True
# except ImportError:
#     print("CodeBLEU not found", flush=True)

check_dependencies()
check_version()
check_nltk_data()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL


def main():
    print("Python script starting...")
    logger.info("Checking logger.info")
    logger.warning("Checking logger.warn")
    logger.error("Checking logger.error")

    torch.cuda.empty_cache()

    model_args, data_args, training_args, adapter_args, advfusion_args = (
        parse_arguments()
    )

    advadp_path_list: list[str] = advfusion_args.advadp_path_list

    setup_logging(logger, training_args.get_process_log_level())

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}\n")
    logger.info(f"Adapter parameters {adapter_args}\n")
    logger.info(f"Data parameters {data_args}\n")
    logger.info(f"Model parameters {model_args}\n")

    if data_args.source_prefix is None and (
        model_args.model_name_or_path.startswith("t5")
        or model_args.model_name_or_path.startswith("codet5")
    ):
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    last_checkpoint = None
    if training_args.do_train:
        last_checkpoint = handle_last_checkpoint(
            training_args.output_dir,
            training_args.overwrite_output_dir,
            training_args.resume_from_checkpoint,
        )

    set_seed(training_args.seed)
    is_decoder_only = "llama" in model_args.model_name_or_path.lower()
    is_gen_job = (
        data_args.train_file is not None
        and "spp" in data_args.train_file.lower()
        or data_args.test_file is not None
        and "spp" in data_args.test_file.lower()
        or data_args.validation_file is not None
        and "spp" in data_args.validation_file.lower()
    )

    raw_datasets = load_raw_datasets(
        data_args.dataset_name,
        data_args.dataset_config_name,
        data_args.train_file,
        data_args.validation_file,
        data_args.test_file,
        training_args.seed,
    )
    logger.info(f"raw_datasets: {raw_datasets}")
    # if training_args.do_train:
    #     logger.info(f"First training sample: {raw_datasets['train'][0]}")
    # if training_args.do_eval:
    #     logger.info(f"First eval sample: {raw_datasets['validation'][0]}")
    # if training_args.do_predict:
    #     logger.info(f"First test sample: {raw_datasets['test'][0]}")

    [h.flush() for h in logger.handlers]

    config = init_config(
        config_name=model_args.config_name,
        model_name_or_path=model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = init_tokenizer(
        tokenizer_name_or_path=model_args.tokenizer_name_or_path,
        model_name_or_path=model_args.model_name_or_path,
        use_fast_tokenizer=model_args.use_fast_tokenizer,
        cache_dir=model_args.cache_dir,
    )

    if model_args.train_tokenizer and model_args.use_fast_tokenizer:
        logger.info("Training tokenizer...")
        training_corpus = get_training_corpus(
            raw_datasets["train"],
            [data_args.text_column, data_args.summary_column],
            1000,
        )
        tokenizer = train_tokenizer(tokenizer, training_corpus)

    model, model_dtype = init_model(
        model_name_or_path=model_args.model_name_or_path,
        model_config=config,
        cache_dir=model_args.cache_dir,
        quantization_mode=model_args.quantization_mode,
        is_decoder_only=is_decoder_only,
    )

    # Convert the model into an adapter model
    adapter_name = None
    if adapter_args.train_adapter:
        if adapter_args.use_adapterhub:
            match adapter_args.adapter_config:
                case "advfusion":
                    (target_adapter_path, target_adapter_name, adapter_name) = (
                        init_ah_advfusion(
                            advadp_path_list=advadp_path_list,
                            advfusion_target=advfusion_args.advfusion_target,
                            model=model,
                            model_dtype=model_dtype,
                        )
                    )
                    advfusion_args.target_adapter_path = target_adapter_path
                    advfusion_args.target_adapter_name = target_adapter_name
                    # advfusion_args.fusion_name = fusion_name

                case _:
                    adapter_name = init_ah_adapter(
                        adapter_config=adapter_args.adapter_config,
                        config_title=model_args.config_title,
                        model=model,
                        model_dtype=model_dtype,
                    )
                    if model_args.preload_adapter:
                        load_ah_adapter(
                            adapter_path=model_args.adapter_path,
                            adapter_name=adapter_name,
                            model=model,
                            set_active=True,
                        )
            # logger.info(f"Active heads: {model.active_head()}")
            logger.info(f"Adapter Summary:\n{model.adapter_summary()}")
        else:
            if model_args.preload_adapter:
                init_and_load_peft_adapter(
                    adapter_path=model_args.adapter_path,
                    config_title=model_args.config_title,
                    model=model,
                    # device=model.device,
                )
            else:
                init_peft_adapter(
                    adapter_config=adapter_args.adapter_config,
                    config_title=model_args.config_title,
                    model=model,
                )

    if is_decoder_only:
        ensure_decoder_only_padding_token(model=model, tokenizer=tokenizer)

    ensure_decoder_start_token(
        model=model, tokenizer=tokenizer, target_lang=data_args.lang
    )

    ensure_embedding_size(
        model=model,
        tokenizer=tokenizer,
        max_source_length=data_args.max_source_length,
    )

    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`."
        )
        return

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(
        data_args.dataset_name, None
    )
    if data_args.text_column is None:
        text_column = (
            dataset_columns[0]
            if dataset_columns is not None
            else column_names[0]
        )
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = (
            dataset_columns[1]
            if dataset_columns is not None
            else column_names[1]
        )
    else:
        summary_column = data_args.summary_column
        # if summary_column not in column_names:
        #     raise ValueError(
        #         f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
        #     )

    prefix = (
        data_args.source_prefix if data_args.source_prefix is not None else ""
    )

    # Preprocessing the datasets.
    # filter_dataset(
    #     raw_datasets,
    #     text_column=text_column,
    #     summary_column=summary_column,
    #     preprocessing_num_workers=1,
    # )

    ensure_multilingual_tokenizer(
        model=model,
        tokenizer=tokenizer,
        target_lang=data_args.lang,
        forced_bos_token=data_args.forced_bos_token,
    )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(
        model, "prepare_decoder_input_ids_from_labels"
    ):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    logger.info(f"Tokenizer pad token: {tokenizer.pad_token}")

    logger.info(f"Model architucture:\n{model}")
    logger.info(f"Model memory footprint:\n{model.get_memory_footprint()}")

    tokenizer.save_pretrained(model_args.tokenizer_name_or_path)
    logger.info(f"Tokenizer saved to {model_args.tokenizer_name_or_path}")

    preprocess_function = (
        get_decoder_only_preprocessor(
            tokenizer=tokenizer,
            text_column=text_column,
            summary_column=summary_column,
            max_source_length=data_args.max_source_length,
            padding=padding,
            is_text_tokenized=data_args.text_tokenized,
            is_summary_tokenized=data_args.summary_tokenized,
            ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
        )
        if is_decoder_only
        else get_encoder_decoder_preprocessor(
            tokenizer=tokenizer,
            text_column=text_column,
            summary_column=summary_column,
            prefix=prefix,
            max_source_length=data_args.max_source_length,
            max_target_length=max_target_length,
            padding=padding,
            is_text_tokenized=data_args.text_tokenized,
            is_summary_tokenized=data_args.summary_tokenized,
            ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
        )
    )

    generation_preprocess_function = get_generation_preprocessor(
        tokenizer=tokenizer,
        text_column=text_column,
        max_source_length=data_args.max_source_length,
        padding=padding,
        is_text_tokenized=data_args.text_tokenized,
        ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
    )

    max_train_samples = data_args.max_train_samples
    train_dataset = None
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples
            )
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(
            desc="train dataset map pre-processing"
        ):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
            logger.info(f"train_dataset:\n{train_dataset}")

        if is_decoder_only:
            # if data_args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > model.config.max_position_embeddings:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    f"Using block_size={min(data_args.max_source_length, model.config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
                )
                if model.config.max_position_embeddings > 0:
                    block_size = min(
                        data_args.max_source_length,
                        model.config.max_position_embeddings,
                    )
                else:
                    block_size = data_args.max_source_length
            # else:
            #     if data_args.block_size > tokenizer.model_max_length:
            #         logger.warning(
            #             f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
            #             f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            #         )
            #     block_size = min(data_args.block_size, tokenizer.model_max_length)

            group_texts = get_text_grouper(block_size)
            with training_args.main_process_first(
                desc="grouping texts together"
            ):
                train_dataset = train_dataset.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
                logger.info(f"train_dataset:\n{train_dataset}")

    max_eval_samples = data_args.max_eval_samples
    eval_dataset = None
    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples
            )
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(
            desc="validation dataset map pre-processing"
        ):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
            logger.info(f"eval_dataset:\n{eval_dataset}")
        if is_decoder_only:
            # if data_args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > model.config.max_position_embeddings:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    f"Using block_size={min(data_args.max_source_length, model.config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
                )
                if model.config.max_position_embeddings > 0:
                    block_size = min(
                        data_args.max_source_length,
                        model.config.max_position_embeddings,
                    )
                else:
                    block_size = data_args.max_source_length
            # else:
            #     if data_args.block_size > tokenizer.model_max_length:
            #         logger.warning(
            #             f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model "
            #             f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            #         )
            #     block_size = min(data_args.block_size, tokenizer.model_max_length)

            group_texts = get_text_grouper(block_size)
            with training_args.main_process_first(
                desc="grouping texts together"
            ):
                eval_dataset = eval_dataset.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
            logger.info(f"eval_dataset:\n{eval_dataset}")

    max_predict_samples = data_args.max_predict_samples
    predict_dataset = None
    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(
            desc="prediction dataset map pre-processing"
        ):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
            logger.info(f"predict_dataset:\n{predict_dataset}")

    if training_args.predict_with_generate:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("Generation requires a test dataset")
        generation_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            generation_dataset = generation_dataset.select(
                range(max_predict_samples)
            )
        with training_args.main_process_first(
            desc="generation dataset map pre-processing"
        ):
            generation_dataset = generation_dataset.map(
                generation_preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on generation dataset",
            )
            logger.info(f"generation_dataset:\n{generation_dataset}")
    # Data collator
    label_pad_token_id = (
        -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )
    data_collator = (
        default_data_collator
        # DataCollatorForLanguageModeling(
        #     tokenizer,
        #     mlm=False,
        # )
        if is_decoder_only
        else DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    )
    metric_bleu = None
    if data_args.metric_path is not None:
        metric_bleu = evaluate.load(path=data_args.metric_path)
    metric_rouge = None
    if data_args.metric_path_alt is not None:
        metric_rouge = evaluate.load(path=data_args.metric_path_alt)

    compute_metrics = get_compute_metrics(
        tokenizer=tokenizer,
        ignore_pad_token_for_loss=data_args.ignore_pad_token_for_loss,
        metric_rouge=metric_rouge,
        metric_bleu=metric_bleu,
        is_decoder_only=is_decoder_only,
        metric_path=data_args.metric_path,
    )

    # Initialize our Trainer
    trainer: Trainer | Seq2SeqTrainer | None = None

    # use_sft = False
    # sft_trainer: SFTTrainer | None = None
    if training_args.do_train or training_args.do_eval:
        if adapter_args.train_adapter:
            trainer_class = (
                AdapterTrainer if is_decoder_only else Seq2SeqAdapterTrainer
            )
        else:
            trainer_class = Trainer if is_decoder_only else Seq2SeqTrainer

        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=(
                preprocess_logits_for_metrics if is_decoder_only else None
            ),
        )

        # if use_sft and is_decoder_only:
        #     response_template_with_context = "\n### Assistant:"
        #     response_template_ids = tokenizer.encode(
        #         response_template_with_context, add_special_tokens=False
        #     )[2:]
        #     sft_data_collator = DataCollatorForCompletionOnlyLM(
        #         response_template_ids, tokenizer=tokenizer
        #     )

        #     sft_args = SFTConfig(
        #         output_dir=training_args.output_dir,
        #     )
        #     sft_trainer = SFTTrainer(
        #         model=model,
        #         args=sft_args,
        #         train_dataset=train_dataset,
        #         eval_dataset=eval_dataset,
        #         tokenizer=tokenizer,
        #         data_collator=sft_data_collator,
        #         compute_metrics=compute_metrics,
        #         preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        #     )

        # Early stopping
        if data_args.patience and data_args.patience > 0:
            logger.info(
                f"metric for choosing best model is {training_args.metric_for_best_model}"
            )
            callback = EarlyStoppingCallback(
                early_stopping_patience=data_args.patience
            )
            trainer.add_callback(callback)
            training_args.load_best_model_at_end = True

    # Training
    if (
        trainer is not None
        and training_args.do_train
        and train_dataset is not None
        and max_train_samples > 0
    ):
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        if adapter_args.adapter_config == "advfusion":
            zero_freeze_adapter(
                model, advfusion_args.target_adapter_name, model_dtype
            )

        timer = CudaTimer()
        timer.start()
        # if sft_trainer is not None:
        #     train_result = sft_trainer.train(resume_from_checkpoint=checkpoint)
        # else:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        if adapter_args.adapter_config == "advfusion":
            handle_metrics(
                trainer=trainer,
                prefix="train_before",
                metrics=train_result.metrics,
                sample_count=max_train_samples,
            )
            unfreeze_reload_adapter(
                model,
                advfusion_args.target_adapter_path,
                advfusion_args.target_adapter_name,
            )
            train_result = trainer.train(resume_from_checkpoint=checkpoint)

        elapsed = timer.stop()
        if elapsed is not None:
            performance_metrics = {}
            performance_metrics.update({"total_gpu_time": elapsed})
            peak_memory = (torch.cuda.max_memory_allocated() / 1024**2) / 1000
            performance_metrics.update({"peak_memory": peak_memory})
            handle_metrics(
                trainer=trainer,
                prefix="performance",
                metrics=performance_metrics,
            )

        if not adapter_args.use_adapterhub:
            # model.save_pretrained(training_args.output_dir)
            trainer.save_model()

        handle_metrics(
            trainer=trainer,
            prefix="train",
            metrics=train_result.metrics,
            sample_count=max_train_samples,
        )

        if adapter_args.train_adapter:
            save_ah_adapter(
                adapter_path=model_args.adapter_path,
                adapter_config=adapter_args.adapter_config,
                adapter_name=adapter_name,
                model=model,
            )

        trainer.save_state()

    # Evaluation
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    # num_beams = (
    #     data_args.num_beams
    #     if data_args.num_beams is not None
    #     else training_args.generation_num_beams
    # )
    if (
        trainer is not None
        and training_args.do_eval
        and eval_dataset is not None
        and max_eval_samples > 0
    ):
        logger.info("*** Evaluate ***")
        metrics = (
            trainer.evaluate(metric_key_prefix="eval")
            if is_decoder_only
            else trainer.evaluate(
                metric_key_prefix="eval",
                max_length=max_length,
                # num_beams=num_beams,
            )
        )
        handle_metrics(
            trainer=trainer,
            prefix="eval",
            metrics=metrics,
            sample_count=max_eval_samples,
        )

    if (
        trainer is not None
        and training_args.do_predict
        and predict_dataset is not None
        and max_predict_samples > 0
    ):
        logger.info("*** Predict ***")

        if not is_decoder_only:
            predict_results = (
                trainer.predict(
                    predict_dataset,
                    metric_key_prefix="predict",
                )
                if is_decoder_only
                else trainer.predict(
                    predict_dataset,
                    metric_key_prefix="predict",
                    max_length=max_length,
                    # num_beams=num_beams,
                )
            )

            handle_metrics(
                trainer=trainer,
                prefix="predict",
                metrics=predict_results.metrics,
                sample_count=max_predict_samples,
            )

            labels = predict_results.label_ids
            preds = predict_results.predictions

            if (
                labels is not None
                and preds is not None
                and trainer.is_world_process_zero()
            ):
                if training_args.predict_with_generate:
                    generation_save_dir = (
                        model_args.generation_output_path
                        if model_args.generation_output_path is not None
                        else training_args.output_dir
                    )
                    generation_from_predict_encoder_decoder(
                        tokenizer=tokenizer,
                        preds=preds,
                        labels=labels,
                        raw_dataset=raw_datasets["test"],
                        tokenized_dataset=predict_dataset,
                        text_column=text_column,
                        summary_column=summary_column,
                        save_path=generation_save_dir,
                    )

        if is_decoder_only and trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                generation_save_dir = (
                    model_args.generation_output_path
                    if model_args.generation_output_path is not None
                    else training_args.output_dir
                )
                generation_decoder_only(
                    model=model,
                    tokenizer=tokenizer,
                    raw_dataset=raw_datasets["test"],
                    text_column=text_column,
                    summary_column=summary_column,
                    max_predict_samples=max_predict_samples,
                    max_source_length=data_args.max_source_length,
                    max_new_tokens=model_args.max_new_tokens,
                    padding=padding,
                    save_path=generation_save_dir,
                    metric_rouge=metric_rouge,
                    metric_bleu=metric_bleu,
                    metric_path=data_args.metric_path,
                )

            num_samples_per_task = data_args.humaneval_num
            if is_gen_job and num_samples_per_task > 0:
                run_humaneval(
                    model=model,
                    tokenizer=tokenizer,
                    num_samples_per_task=num_samples_per_task,
                    output_dir=training_args.output_dir,
                )
        # if trainer.is_world_process_zero():
        #     source = raw_datasets["test"].select(
        #         # range(min(max_predict_samples, len(predict_dataset)))
        #         range(10)
        #     )
        #     for i, inputs in enumerate(source):
        #         inputs = "summarize: " + " ".join(inputs["code_tokens"])
        #         logger.info(f"Input: {inputs}")
        #         inputs = tokenizer.encode(inputs, return_tensors="pt").to(model.device)
        #         outputs = model.generate(inputs)
        #         outputs = tokenizer.decode(outputs[0])
        #         logger.info(f"Output: {outputs}")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
