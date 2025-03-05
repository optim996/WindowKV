import torch
from torch import nn


class BertClassifier(nn.Module):
    def __init__(self, model, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = model
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        
        pooled_output = pooled_output.to(self.bert.device)
        
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        output = self.relu(linear_output)
        
        return output
    

def classifier(input_string, bert_tokenizer, bert_model, device):

    with torch.no_grad():
        inputs = bert_tokenizer(input_string, return_tensors="pt", truncation=False)
        input_ids = inputs["input_ids"].squeeze(0)

        if len(input_ids) >= 512:
            truncated_ids = torch.cat([input_ids[:256], input_ids[-256:]])
        else:
            truncated_ids = input_ids

        input_ids = truncated_ids.unsqueeze(0).to(device)  
        attention_mask = torch.ones_like(input_ids).to(device)

        bert_model = bert_model.to(device)

        output = bert_model(input_ids, attention_mask)

        pred = output.argmax(dim=1)
        
        if pred.item() == 0:
            category = "qa"
        elif pred.item() == 1:
            category = "other"
        else:
            category = "unknown"
    
    return category