# go

from dataset.train_test_data import train_test_data
X_train, X_val, X_test, y_train, y_val, y_test = train_test_data()
from collections import Counter
def build_text(texts,vocab_size=10000):
    counter = Counter()
    for word in texts:
          counter.update(word.split())
    vocab = {'<PAD>':0,'<UNK>':1}
    for word , _ in counter.most_common(vocab_size-2):
          vocab[word] = len(vocab)
    return vocab       
vocab = build_text(X_train)
vocab_size = len(vocab)
def encode_text(texts,vocab):
    return [vocab.get(word,vocab['<UNK>'])for word in texts.split()]
def pad_seq(seq,max_len=50):
    if len(seq) > max_len:
        return seq[:max_len]
    return seq + [0]*(max_len-len(seq))
X_train_seq = [pad_seq(encode_text(texts,vocab))for texts in X_train]
X_test_seq = [pad_seq(encode_text(texts,vocab))for texts in X_test]
X_val_seq = [pad_seq(encode_text(texts,vocab))for texts in X_val]
import torch 
import torch.nn as nn
X_train_tensor = torch.tensor(X_train_seq,dtype=torch.long)
X_test_tensor = torch.tensor(X_test_seq,dtype=torch.long)
X_val_tensor = torch.tensor(X_val_seq,dtype=torch.long)
y_train_tensor = torch.tensor(y_train.values,dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values,dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values,dtype=torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TransformerClassifier(nn.Module):
    def __init__(self,vocab_size,embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4, # self attention (multi head)
            dim_feedforward=256, # for learning
            batch_first=True  
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.classifier = nn.Linear(embed_dim,1)
    def forward(self,x):
        pad_mask = (x==0) # True where PAD
        embed = self.embedding(x)
        encoded = self.encoder(embed,src_key_padding_mask=pad_mask)
        after_padding = encoded.mean(dim=1)
        return self.classifier(after_padding)    
model = TransformerClassifier(vocab_size).to(device)
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
from torch.utils.data import TensorDataset , DataLoader
train_loader = DataLoader(TensorDataset(X_train_tensor,y_train_tensor),batch_size=32,shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor,y_val_tensor),batch_size=32,shuffle=False)
def train_one_epoch(model,data_loader,loss_function,optimizer,device):
    total , correct , total_loss = 0,0,0
    model.train()

    for X , y in data_loader:
        X = X.to(device)
        y = y.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(X)
        loss = loss_function(logits,y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += y.shape[0]
        preds = (logits>=0).float()
        correct += (preds==y).sum().item()

    return total_loss/len(data_loader) , correct/total
@torch.no_grad()
def validation_one_epoch(model,data_loader,loss_function,device):
    total , correct , total_loss = 0,0,0
    model.eval()

    for X , y in data_loader:
        X = X.to(device)
        y = y.to(device).unsqueeze(1)

        
        logits = model(X)
        loss = loss_function(logits,y)


        total_loss += loss.item()
        total += y.shape[0]
        preds = (logits>=0).float()
        correct += (preds==y).sum().item()

    return total_loss/len(data_loader) , correct/total
class EarlyStopping:
    def __init__(self,patience=3):
        self.patience = patience
        self.Counter = 0
        self.best_state = None
        self.best_loss = float('inf')

    def step(self,val_loss,model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.Counter = 0
            self.best_state = model.state_dict()
            return False
        else:
            self.Counter += 1
            return self.Counter >= self.patience   
early_stopping = EarlyStopping(patience=3)

for epoch in range(10):
    train_loss , train_acc = train_one_epoch(model,train_loader,loss_function,optimizer,device)
    val_loss , val_acc = validation_one_epoch(model,val_loader,loss_function,device)

    print(f"epoch:{epoch} , train_acc: {train_acc} , val_acc: {val_acc}")

    if early_stopping.step(val_loss,model):
        print('EarlyStopping Triggred')
        break
