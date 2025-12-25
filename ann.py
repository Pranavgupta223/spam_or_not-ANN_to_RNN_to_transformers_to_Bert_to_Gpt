from dataset.train_test_data import train_test_data

X_train ,X_test , y_train,y_test = train_test_data()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000,ngram_range=(1,2))
X_train_tdidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

import torch

X_train_tensor = torch.from_numpy(X_train_tdidf.toarray()).float()
X_test_tensor = torch.from_numpy(X_test_tfidf.toarray()).float()
y_train_tesor = torch.tensor(y_train.values,dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values,dtype=torch.float32)

from sklearn.model_selection import train_test_split
X_tr,X_val,y_tr,y_val = train_test_split(
    X_test_tensor,y_test_tensor,test_size=0.3,random_state=42,stratify=y_test_tensor
)

from torch.utils.data import TensorDataset , DataLoader

train_ds = TensorDataset(X_tr,y_tr)
val_ds = TensorDataset(X_val,y_val)

train_loader = DataLoader(train_ds,batch_size=32,shuffle=True)
val_loader = DataLoader(val_ds,batch_size=32,shuffle=False)

import torch.nn as nn

class ANN(nn.Module):
    
    def __init__(self,input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim,128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64,1)
        )

    def forward(self,x):
        return self.model(x)   
             
model = ANN(input_dim=X_train_tensor.shape[1])

loss_function = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

def train_one_epoch(model,loader,loss_function,optimizer):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for x,y in loader:
        y = y.unsqueeze(1)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_function(logits,y)

        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        # accuracy logic
        preds = (logits>=0).float()
        correct += (preds==y).sum().item()
        total += y.shape[0]

    return total_loss/len(loader) , correct/total

def validation_one_epoch(model,loader,loss_function):
    model.eval()
    total_loss , correct , total = 0,0,0

    for x,y in loader:
        y = y.unsqueeze(1)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_function(logits,y)
        
        total_loss += loss.item()
        preds = (logits>=0).float()
        correct += (preds==y).sum().item()

        total += y.shape[0]

    return total_loss / len(loader),correct/total    
        
class EarlyStopping:
    def __init__(self,patience=3):
        self.patience = patience
        self.counter = 0
        self.best_state = None
        self.best_loss = float('inf')

    def step(self,val_loss,model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_state = model.state_dict()
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter>=self.patience    
        
early_stopping = EarlyStopping(patience=3)

for epoch in range(50):
    train_loss , train_accuracy = train_one_epoch(model,train_loader,loss_function,optimizer)
    val_loss , val_accuracy = validation_one_epoch(model,val_loader,loss_function)

    print(f"epoch-> {epoch+1}")
    print(f"train_loss: {train_loss:.4f} || train_accuracy: {train_accuracy:.4f}")
    print(f"val_loss: {val_loss:.4f} || val_accuracy: {val_accuracy:.4f}")
    
    if early_stopping.step(val_loss,model):
        print("Early_stopping_triggered")
        break
