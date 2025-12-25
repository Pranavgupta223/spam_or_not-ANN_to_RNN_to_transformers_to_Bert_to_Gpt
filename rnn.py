# go


from dataset.train_test_data import train_test_data
X_train , X_val , X_test , y_train , y_val , y_test = train_test_data()
# RNN ( recurrent neural network)
from collections import Counter
counter = Counter()

def build_vocab(texts,vocab_size=10000):
    for text in texts:
        counter.update(text.split())

    vocab = {'<PAD>':0,"<UNK>":1}    
    for word , _ in counter.most_common(vocab_size-2):
        vocab[word] = len(vocab)
    return vocab    
def encode_text(text,vocab):
    return [vocab.get(word,vocab["<UNK>"]) for word in text.split()]
vocab = build_vocab(X_train)
vocab_size = len(vocab)
def pad_sequence(text,max_len):
    if len(text)>max_len:
        return text[:max_len]
    return text + [0]*(max_len - len(text))
X_train_seq = [pad_sequence(encode_text(text,vocab),max_len=50) for text in X_train ]
X_test_seq = [pad_sequence(encode_text(text,vocab),max_len=50) for text in X_test]
X_val_seq = [pad_sequence(encode_text(text,vocab),max_len=50)for text in X_val]
import torch
X_train_tensor = torch.tensor(X_train_seq,dtype=torch.long)
X_test_tensor = torch.tensor(X_test_seq,dtype=torch.long)
X_val_tensor = torch.tensor(X_val_seq,dtype=torch.long)
y_train_tensor = torch.tensor(y_train.values,dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values,dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values,dtype=torch.float32)
import torch.nn as nn
class RNNModel(nn.Module):
    def __init__(self,vocab_size,embed_size=128,hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_size,padding_idx=0)
        self.rnn = nn.RNN(embed_size,hidden_size,batch_first=True)
        self.fc = nn.Linear(hidden_size,1)

    def forward(self,x):
        embd = self.embedding(x)
        _ , h_n = self.rnn(embd)
        out = self.fc(h_n.squeeze(0))
        return out    
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = RNNModel(vocab_size)
model = model.to(device)
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
def train_one_epoch(model,data_loader,loss_function,optimizer):
    model.train()
    total_loss = 0
    total = 0
    correct = 0

    for X , y in data_loader:
        X = X.to(device)
        y = y.to(device).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(X)
        loss = loss_function(logits,y)

        loss.backward()
        optimizer.step()

        total += y.shape[0]
        total_loss += loss.item() 

        preds = (logits>=0).float()
        correct += (preds == y).sum().item()

    return total_loss/len(data_loader) , correct/total       
with torch.no_grad():
  def test_val_data(model,data_loader,loss_function):
    model.eval()
    total = 0
    total_loss = 0
    correct = 0

    for X , y in data_loader:
        X = X.to(device)
        y = y.to(device).unsqueeze(1)

        logits = model(X)
        loss = loss_function(logits,y)

        total += y.shape[0]
        total_loss += loss.item()

        preds = (logits>=0).float()
        correct += (preds == y).sum().item()

    return total_loss / len(data_loader) , correct/total    
class EarlyStopping:
    def __init__(self,patience=3):
        self.patience = patience
        self.counter = 0
        self.best_state = None
        self.best_loss = float("inf")

    def step(self,val_loss,model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = model.state_dict()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience        
from torch.utils.data import TensorDataset , DataLoader

train_dataset = TensorDataset(X_train_tensor,y_train_tensor)
val_dataset = TensorDataset(X_val_tensor,y_val_tensor)

train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
val_loader = DataLoader(val_dataset , batch_size= 32 , shuffle=False)
early_stopping = EarlyStopping()

for epoch in range(50):
    trian_loss , train_acc = train_one_epoch(model,train_loader,loss_function,optimizer)
    val_loss , val_acc = test_val_data(model,val_loader,loss_function)

    print(f"epoch:{epoch} , train_loss :{trian_loss} , val_loss : {val_loss}")
    print(f"epoch:{epoch} , train_acc :{train_acc} , val_acc : {val_acc}")

    if early_stopping.step(val_loss,model):
        print("EarlyStopping_triggred")
        break
     