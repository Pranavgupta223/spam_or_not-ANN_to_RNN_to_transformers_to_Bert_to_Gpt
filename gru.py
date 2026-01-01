from dataset.train_test_data import train_test_data
X_train , X_test ,X_val , y_train,y_test,y_val = train_test_data() 
from collections import Counter
def build_text(texts,vocab_size=10000):
    counter = Counter()
    for word in texts:
        counter.update(word.split())
    vocab = {"<PAD>":0,"<UNK>":1}  
    for word , _ in counter.most_common(vocab_size-2):
        vocab[word] = len(vocab)
    return vocab      
vocab = build_text(X_train)
vocab_size = len(vocab)
def encode_text(texts,vocab):
    return [ vocab.get(word,vocab['<UNK>'])for word in texts]
def pad_seq(seq,max_len = 50):
    if len(seq) > max_len:
        return seq[:max_len]
    return seq + [0]*(max_len-len(seq))
X_train_seq = [pad_seq(encode_text(texts,vocab))for texts in X_train]
X_val_seq = [pad_seq(encode_text(texts,vocab))for texts in X_val]
import torch 
from torch.utils.data import TensorDataset , DataLoader
import torch.nn as nn
X_train_tensor = torch.tensor(X_train_seq,dtype=torch.long)
X_val_tensor = torch.tensor(X_val_seq,dtype=torch.long)

y_train_tensor = torch.tensor(y_train.values , dtype=torch.long)
y_val_tensor = torch.tensor(y_val.values , dtype=torch.long)
train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
val_dataset = TensorDataset(X_val_tensor,y_val_tensor)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
val_loader = DataLoader(val_dataset , batch_size=32,shuffle=False)
class GRUModel(nn.Module):
    def __init__(self,vocab_size,embed_size=128,hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_size,padding_idx=0)
        self.gru = nn.GRU(embed_size,hidden_size,batch_first=True)
        self.fc = nn.Linear(hidden_size,1)

    def forward(self,x):
        embd = self.embedding(x)
        _ , h_n = self.gru(embd)
        out = self.fc(h_n.squeeze(0))
        return out    
model = GRUModel(vocab_size=vocab_size)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
loss_function = nn.BCEWithLogitsLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train_one_epoch(model,data_loader,loss_function,optimizer,device):
    model.to(device).train()
    total , correct , total_loss = 0 , 0 , 0 

    for X , y in data_loader:
        X = X.to(device)
        y = y.to(device).float().unsqueeze(1)

        optimizer.zero_grad()
        logits = model(X)
        loss = loss_function(logits,y)

        loss.backward()
        optimizer.step()

        total += y.shape[0]
        total_loss += loss.item()
        preds = (logits >= 0).float()
        correct += (preds == y).sum().item()
    return total_loss/len(data_loader) , correct/total

def validation_one_epoch(model,data_loader,loss_function,device):
    
    model.to(device).eval()
    total , correct , total_loss = 0 , 0 , 0 

    for X , y in data_loader:
        X = X.to(device)
        y = y.to(device).float().unsqueeze(1)
  
        logits = model(X)
        loss = loss_function(logits,y)

        total += y.shape[0]
        total_loss += loss.item()
        preds = (logits >= 0).float()
        correct += (preds == y).sum().item()
    return total_loss/len(data_loader) , correct/total
class EarlyStopping:
    def __init__(self,patience=3):
        self.patience = patience
        self.Counter = 0
        self.best_loss = float('inf')
        self.best_state = None


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

    print(f"epoch : {epoch} train_acc: {train_acc} , val_acc: {val_acc}")
    
    if early_stopping.step(val_loss , model):
        print('EarlyStopping Trigrred')
        break
     
