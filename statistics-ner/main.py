import os
import json
import torch
import numpy as np
from config import NerConfig
from  model.roberta_bilstm_crf import BertNer
from data_loader import NerDataset
from tqdm import tqdm
from seqeval.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_to_txt(text:str, path:str):
    with open(path, 'a', encoding="utf-8") as file:
        file.write(text + '\n')

class Trainer:
    def __init__(self, args, model, train_loader, val_loader, optimizer, scheduler, total_step):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = args.epochs
        self.output_dir = args.output_dir
        self.id2label = args.id2label
        self.save_step = args.save_step
        self.total_step = total_step

    def train(self):
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for step, batch_data in tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=True):
                for key, value in batch_data.items():
                    batch_data[key] = value.to(device)
                input_ids = batch_data["input_ids"]
                attention_mask = batch_data["attention_mask"]
                labels = batch_data["labels"]
                output = self.model(input_ids, attention_mask, labels)
                loss = output.loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # PS：关于scheduler是epoch结束后进行step还是batch结束后step，在下不太清楚，有待后续讨论
                self.scheduler.step()
                if step % self.save_step == 0 and step > 1:
                    loss_report = f'epoch{epoch}-setp{step}-loss:{loss.item()}'
                    print(loss_report + "-model will be save soon")
                    save_to_txt(loss_report, os.path.join(self.output_dir, "loss_for_step.txt"))
                    torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"checkpoint_{step}.bin"))
                    # PS：当你进行断点训练时，切记同时保存optimizer和scheduler的参数
                    # 将self.model.state_dict()替换为{"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict()}
                    # 断点训练导入模型的方法相信大家都会，这里不做赘述
                
            # 进行验证，目前默认是一个epoch验证一次，当然如果数据量足够大，建议基于step进行验证
            # PS:这一段写得比较乱，效率很低，希望后续能够进行优化
            self.model.eval()
            preds = []
            trues = []
            with torch.no_grad():
                for step, batch_data in tqdm(enumerate(self.val_loader), total=len(self.val_loader), leave=True):
                    for key, value in batch_data.items():
                        batch_data[key] = value.to(device)
                    input_ids = batch_data["input_ids"]
                    attention_mask = batch_data["attention_mask"]
                    labels = batch_data["labels"]
                    output = self.model(input_ids, attention_mask, labels)
                    logits = output.logits
                    attention_mask = attention_mask.detach().cpu().numpy()
                    labels = labels.detach().cpu().numpy()
                    batch_size = input_ids.size(0)
                    for i in range(batch_size):
                        length = sum(attention_mask[i])
                        logit = [self.id2label[j] for j in logits[i][1:length]]
                        label = [self.id2label[j] for j in labels[i][1:length]]
                        preds.append(logit)
                        trues.append(label)
                f1 = f1_score(trues, preds)
                acc = accuracy_score(trues, preds)
                r = recall_score(trues, preds)
                p = precision_score(trues, preds)
                eval_report = f'epoch{epoch}-pre-recall-f1-acc:{p},{r},{f1},{acc}'
                print(classification_report(trues, preds))
                save_to_txt(eval_report, os.path.join(self.output_dir, "f1_acc_for_epoch.txt"))
                # PS1：可以根据evaluate的loss设计early stop，这里偷懒了没写，算法大神保佑，训必达！
                # PS2：可以根据evaluate的loss或指标，保存验证效果最好的模型，虽然这么做有些不合规范
        # 保存一个最终的模型，防止兄弟们save_batch设置太大
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"checkpoint_final.bin"))
                
def build_optimizer_and_scheduler(args, model, t_total):
    # 检查模型是否有module属性，当使用DataParallel或DistributedDataParallel时，model会被添加module属性
    module = (model.module if hasattr(model, "module") else model)
    # 定义不需要权重衰减的参数名
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    model_param = list(module.named_parameters())
    bert_param_optimizer = []
    other_param_optimizer = []
    for name, para in model_param:
        space = name.split('.')
        if space[0] == 'bert_module' or space[0] == "bert":
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))
    # bert模型和其他的层初始设置不同的学习率和权重衰减
    optimizer_grouped_parameters = [
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.bert_learning_rate},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.bert_learning_rate},
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay, 'lr': args.learning_rate},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args.learning_rate}]
    # 定义优化器和调度器
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.bert_learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * t_total), num_training_steps=t_total)
    return optimizer, scheduler

def main():
    # 导入参数设置的同时将其进行保存
    args = NerConfig()
    with open(os.path.join(args.output_dir, "ner_args.json"), "w") as fp:
        json.dump(vars(args), fp, ensure_ascii=False, indent=2)

    # 导入tokenizer和数据以构建数据集
    tokenizer = BertTokenizer.from_pretrained(args.bert_dir)
    with open(args.train_data_path, 'r', encoding='utf-8') as file:
        train_data = json.load(file)
    with open(args.val_data_path, 'r', encoding='utf-8') as file:
        val_data = json.load(file)
    train_dataset = NerDataset(train_data, args, tokenizer)
    val_dataset = NerDataset(val_data, args, tokenizer)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.val_batch_size, num_workers=args.num_workers)

    # 创建模型并置入device中
    model = BertNer(args)
    model.to(device)
    # 数据并行下的多卡训练，有卡的同学可以取消该注释
    # model = torch.nn.DataParallel(model)

    # 训练及验证
    total_step = len(train_loader) * args.epochs
    optimizer, scheduler = build_optimizer_and_scheduler(args, model, total_step)
    train = Trainer(args=args, model=model, train_loader=train_loader,val_loader=val_loader, 
                    optimizer=optimizer, scheduler=scheduler, total_step=total_step)
    train.train()


if __name__ == "__main__":
    main()
