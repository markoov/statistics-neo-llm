#!/bin/bash
# 参数如下
# cutoff_len：即输入序列分词后的最大长度，默认为1024
# gradient_accumulation_steps：梯度累计步数，默认为8
# loraplus_lr_ratio：LoRA+ 中 B 矩阵的学习率倍数，默认为16.0
# optim：使用的优化器，如paged_adamw_8bit
# max_samples：每个数据集的最大样本数，默认为3000
# quantization_bit：启用 4/8 比特模型量化
# compute_type：是否使用混合精度训练
# val_size：验证集占全部样本的百分比
# logging_steps：每两次日志输出间的更新步数
# neftune_alpha：嵌入向量所添加的噪声大小
# resize_vocab：更改分词器词表和嵌入层的大小
# upcast_layernorm：缩放归一化层
# num_layer_trainable：可训练模型层的数量
# lora_rank：LoRA矩阵的秩大小，默认为8
# lora_alpha：LoRA缩放系数大小，默认为16


CUDA_VISIBLE_DEVICES=0 python /data/LLaMA-Factory/src/train_bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path /data/ChatGLM3/chatglm3-6b \
    --dataset ruozhiba \
    --dataset_dir /data/LLaMA-Factory/data \
    --template chatglm3 \
    --finetuning_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_target query_key_value \
    --output_dir /data/model-train/lora-model-1 \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 20 \
    --warmup_steps 20 \
    --save_steps 200 \
    --eval_steps 200 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 2e-4 \
    --num_train_epochs 10.0 \
    --max_samples 4096 \
    --val_size 0.1 \
    --plot_loss \
    --fp16
