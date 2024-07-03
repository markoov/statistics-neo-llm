import torch.nn as nn
from .torchcrf import CRF
from transformers import BertModel, BertConfig

class ModelOutput:
    def __init__(self, logits, labels, loss=None):
        self.logits = logits
        self.labels = labels
        self.loss = loss

class BertNer(nn.Module):
    def __init__(self, args):
        super(BertNer, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_dir)
        self.bert_config = BertConfig.from_pretrained(args.bert_dir)
        self.hidden_size = self.bert_config.hidden_size
        self.lstm_hidden = args.lstm_hidden
        self.max_seq_len = args.max_seq_len
        self.bilstm = nn.LSTM(self.hidden_size, self.lstm_hidden, 2, bidirectional=True, batch_first=True, dropout=args.lstm_drop_rate)
        self.linear = nn.Linear(self.lstm_hidden*2, args.num_labels)
        self.crf = CRF(args.num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels):
        # PS1：关于loss是否在forward内进行计算也是一个需要考虑的问题，这里为方便起见，在forward内计算
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # PS2：为了方便初学的同学理解模型，在下将每一层的输入与输出格式写出了，但是根据模型和参数的不同，输入与输出的维度也会变化，望注意！
        seq_out = bert_output[0]    # 这一步seq_out的维度为(batchsize, max_len, 768)
        seq_out, _ = self.bilstm(seq_out)   # 这一步seq_out的维度为(batchsize, max_len, lstm_hidden*2)
        seq_out = self.linear(seq_out)  # 这一步seq_out的维度为(batchsize, max_len, num_labels)
        logits = self.crf.decode(seq_out, mask=attention_mask.bool())
        # 只有在训练和验证时才计算损失
        if labels is not None:
            loss = -self.crf.forward(seq_out, labels, mask=attention_mask.bool(), reduction='mean')
        else:
            loss = None
        model_output = ModelOutput(logits, labels, loss)
        return model_output