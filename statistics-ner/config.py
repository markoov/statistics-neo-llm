import os
import torch
import json
from transformers import BertTokenizer

class NerConfig:
    def __init__(self):
        # 设置数据集路径
        self.train_data_path = "data/train_data.json"
        self.val_data_path = "data/val_data.json"
        # data_name为数据集名称，可以随意指定
        self.data_name = "statistics"
        # bert_dir为bert模型的路径，默认为chinese-roberta-wwm-ext，需要预先将模型文件下载至model-hub文件夹内
        self.bert_dir = "model-hub/your_encoder_model"
        # output_dir为保存checkpoint模型参数的文件夹路径
        self.output_dir = os.path.join("checkpoint", self.data_name)
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        # 导入label.txt并制作标签
        with open("labels.txt", "r", encoding='utf-8') as f:
            self.labels = f.read().strip().split("\n")
        self.bio_labels = ["O"]
        for label in self.labels:
            self.bio_labels.append("B-{}".format(label))
            self.bio_labels.append("I-{}".format(label))
        self.num_labels = len(self.bio_labels)
        self.label2id = {label: i for i, label in enumerate(self.bio_labels)}
        self.id2label = {i: label for i, label in enumerate(self.bio_labels)}
        print(self.bio_labels)
        # 以下为模型和训练时所用的超参数
        # PS：仅为一个示例，有条件者可以使用其他策略进行参数的微调
        # max_seq_len：输入的最大长度，由于bert的缺陷，最大为512
        # eopch & batch_size $ num_workers：量力而行，money is all you need
        self.max_seq_len = 512
        self.epochs = 10
        self.train_batch_size = 48
        self.val_batch_size = 48
        self.num_workers = 2
        # 其他参数：在下才疏学浅，请各位自便
        # PS：learning_rate不宜过小，因为涉及linear层更新，太小了(3e-4)效果会差得让你怀疑人生
        # PS：实验发现，bert_learning_rate较大(3e-4)的时候，会出现更令人匪夷所思的情况，反正我是挺无语的
        self.lstm_hidden = 128
        self.lstm_drop_rate = 0.3
        self.bert_learning_rate = 3e-5
        self.learning_rate = 3e-3
        self.adam_epsilon = 1e-8
        self.weight_decay = 0.01
        self.warmup_proportion = 0.01
        self.save_step = 500
