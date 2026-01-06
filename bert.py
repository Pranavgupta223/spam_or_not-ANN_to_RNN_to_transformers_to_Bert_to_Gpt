# BERT MODEL

# import 
import torch
import torch.nn as nn
from transformers import BertTokenizer , BertForSequenceClassification
from torch.utils.data import DataLoader,Dataset
from torch.optim import AdamW
from dataset.train_test_data import train_test_data
X_train , X_val , X_test , y_train , y_val , y_test = train_test_data()
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased",num_labels=2).to(device)
Max_len = 64
def encode_text(texts):
    return tokenizer(
        list(texts),
        padding='max_length',
        max_length = Max_len ,
        return_tensors='pt',
        truncation = True
    )
train_enc = encode_text(X_train)
test_enc = encode_text(X_test)
val_enc = encode_text(X_val)
y_train_tensor = torch.tensor(y_train.values , dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values , dtype=torch.long)
y_val_tensor = torch.tensor(y_val.values , dtype=torch.long)
class SpamDataset(Dataset):
    def __init__(self,encoding,labels):
        self.encoding = encoding
        self.labels = labels

    def __getitem__(self,idx):
        item = {k: v[idx] for k,v in self.encoding.items()}   
        item['labels'] = self.labels[idx] 
        return item

    def __len__(self):
        return len(self.labels)    
train_loader = DataLoader(
    SpamDataset(train_enc,y_train_tensor),
    batch_size=4,
    shuffle=True
)

test_loader = DataLoader(
    SpamDataset(test_enc,y_test_tensor),
    batch_size=4,
    shuffle=False
)

val_loader = DataLoader(
    SpamDataset(val_enc,y_val_tensor),
    shuffle=False,
    batch_size=4
)
optimizer = AdamW(model.parameters(),lr=2e-5)
for epoch in range(3):
    total_loss = 0
    model.train()
    i = 1
    for batch in train_loader:
        optimizer.zero_grad()
        batch = {k:v.to(device) for k,v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        i+=1
    print(f'batch {i} | train_loss {total_loss:.4f}')
for epoch in range(3):
    total , correct , total_loss = 0,0,0
    model.eval()
    i = 1
    for batch in val_loader:
        batch = {k:v.to(device) for k,v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        total += batch['labels'].shape[0]
        preds = torch.argmax(outputs.logits,dim=1)
        correct += (preds==batch['labels']).sum().item()
        total_loss = loss.item()
        i += 1
    print(f'batch{i} | val_loss {total_loss/len(batch)} | accuracy {correct/total}')    
