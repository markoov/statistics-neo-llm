import torch
import numpy as np

from torch.utils.data import Dataset


class NerDataset(Dataset):
    def __init__(self, data, args, tokenizer):
        self.data = data
        self.args = args
        self.tokenizer = tokenizer
        self.label2id = args.label2id
        self.max_seq_len = args.max_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        # PS1：如果在您的数据集中，文本项与标签项不为text和label，请在此处进行修改
        # PS2：本方案的设计初衷，在数据处理阶段，就已经将输入数据的长度限制在512以内，但为了项目的可扩展性，仍然设计在数据导入时进行了数据切割
        # PS3：数据切割的方式很暴力，直接取前max_seq_len-2个字符，针对该问题，其实有更好的解决方法，如pooling&concat等，这里不做阐述
        # PS4：dataloader部分数据加载将占用大部分的时间，本人能力受限，暂无办法对该部分进行优化，希望后续进行更多讨论
        text = self.data[item]["text"]
        labels = self.data[item]["label"]
        if len(text) > self.max_seq_len - 2:
            text = text[:self.max_seq_len - 2]
            labels = labels[:self.max_seq_len - 2]
        tmp_input_ids = self.tokenizer.convert_tokens_to_ids(["[CLS]"] + text + ["[SEP]"])
        attention_mask = [1] * len(tmp_input_ids)
        input_ids = tmp_input_ids + [0] * (self.max_seq_len - len(tmp_input_ids))
        attention_mask = attention_mask + [0] * (self.max_seq_len - len(tmp_input_ids))
        labels = [self.label2id[label] for label in labels]
        labels = [0] + labels + [0] + [0] * (self.max_seq_len - len(tmp_input_ids))
        
        input_ids = torch.tensor(np.array(input_ids))
        attention_mask = torch.tensor(np.array(attention_mask))
        labels = torch.tensor(np.array(labels))

        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return data