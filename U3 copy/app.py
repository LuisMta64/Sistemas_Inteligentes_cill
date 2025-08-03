import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def prepare_data(name_file):
    global X_train, X_test, X_val, y_train, y_test, y_val, scaler
    data = pd.read_csv(name_file)
    print(data.info())

    # Convertir Academic_Performance a etiqueta binaria
    data['Target'] = data['Academic_Performance'].apply(lambda x: 1 if x >= 75 else 0)

    data = data[['Daily_Usage_Hours', 'Sleep_Hours', 'Target']]
    X = data.iloc[:, 0:2].values
    y = data['Target'].values  # <- etiquetas ahora son 0 o 1

    scaler = MinMaxScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=0)


prepare_data('teen_phone_addiction_dataset.csv')
class ClassificationNM(nn.Module):
    def __init__ (self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
        
    def mytrain(self, X_train, y_train, X_val, y_val, loss_fn, optimizer):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        epochs = 3000
        X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
        X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_val = torch.tensor(y_val, dtype=torch.float32, device=device)
        
        for epoch in range(epochs):
            #Training
            self.train()
            
            #Forward pass
            y_logist = self(X_train).squeeze()
            y_pred = torch.round(torch.sigmoid(y_logist))
            
            #Calculate loss and accuracy
            loss = loss_fn(y_logist, y_train)
            
            #Optimizer zero grad
            optimizer.zero_grad()
            
            #Loss backward
            loss.backward()
            
            #Optimizer step
            optimizer.step()
            
            #Testing
            self.eval()
            with torch.inference_mode():
                #Forward pass
                val_logists = self(X_val).squeeze()
                val_preds = torch.round(torch.sigmoid(val_logists))
                
                #Calculate loss
                test_loss = loss_fn(val_logists, y_val)
                
                
                if epoch % 100 == 0:
                    y_true_train = y_train.to('cpu').numpy()
                    y_pred_train = y_pred.detach().to('cpu').numpy()
                    y_true_val = y_val.to('cpu').numpy()
                    y_pred_val = val_preds.detach().to('cpu').numpy()
                    
                    acc = accuracy_score(y_true = y_true_train, y_pred = y_pred_train)
                    
                    test_acc = accuracy_score(y_true = y_true_val, y_pred = y_pred_val)
                    
                    print(f"Epoch: {epoch + 100} | Loss: {loss:.5f} | Train Acc: {acc:.5f} | Val loss: {test_loss:.5f} | Val Acc: {test_acc:.5f}")
                
ClassificationNM1 = ClassificationNM().to(device)


loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params= ClassificationNM1.parameters(), lr=0.1)
ClassificationNM1.mytrain(X_train, y_train, X_val, y_val, loss_fn, optimizer)


X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
test_logists = ClassificationNM1(X_test).squeeze()
test_pred = torch.round(torch.sigmoid(test_logists))

y_pred_test = test_pred.detach().to('cpu').numpy()

conf_mat = confusion_matrix(y_test, y_pred_test)

# Etiquetas personalizadas según tu caso
disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=['Low Performance', 'High Performance'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()