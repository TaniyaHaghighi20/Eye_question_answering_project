# Eye_question_answering_project
This is a research on ophthalmic question answering systems using large language models.

# pre-training llama2
deepspeed run_clm.py --deepspeed ds_config.json --model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --do_train \
    --do_eval \
    --num_train_epochs=10 \
    --logging_steps=1 \
    --save_steps=400 \
    --overwrite_output_dir \
    --output_dir models/llama2-7b-chat \
    --low_cpu_mem_usage --preprocessing_num_workers 5 \
    --learning_rate 2e-6 \
    --weight_decay 1e-4 \
    --gradient_accumulation_steps 2 \
    --block_size 512 \
    --warmup_steps 600 \
    --save_total_limit 10 \
    --metric_for_best_model "eval_loss" \
    --lr_scheduler_type "cosine" \
    --logging_dir models/llama2-7b-chat/logs \
    --load_best_model_at_end \
    --evaluation_strategy "steps" \
    --eval_steps 400 \
    --dataloader_num_workers 4 \
    --max_grad_norm 1.0

# sft
deepspeed sft.py --deepspeed ds_config.json