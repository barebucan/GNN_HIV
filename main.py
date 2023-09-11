from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from dataset import MyOwnDataset
import config
import torch
import numpy as np
from utils import calculate_metrics
from Net import GNN
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_one_epoch(epoch, loader, model, optimezer, loss_fn, scheduler):

    all_preds = []
    all_labels = []

    running_loss = 0

    for _, batch in enumerate(tqdm(loader)):
        
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        
        loss = loss_fn(pred, batch.y.float())
        loss.backward()

        optimezer.step()
        optimezer.zero_grad()

        running_loss += loss.item()

        all_preds.append(np.rint(torch.sigmoid(pred).detach().cpu().numpy()))
        all_labels.append(batch.y.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    print(np.sum(all_labels))
    print(np.sum(all_preds))
    calculate_metrics(all_preds, all_labels)

    return running_loss / len(loader)

def validation(epoch, loader, model, loss_fn):

    print(f"Doing validataion for {epoch}...")
    model.eval()
    with torch.no_grad():
        all_preds = []
        all_labels = []

        running_loss = 0

        for _, batch in enumerate(tqdm(loader)):
            
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            
            loss = loss_fn(pred, batch.y.float())
            
            running_loss += loss.item()

            all_preds.append(np.rint(torch.sigmoid(pred).detach().cpu().numpy()))
            all_labels.append(batch.y.detach().cpu().numpy())

        all_preds = np.concatenate(all_preds).ravel()
        all_labels = np.concatenate(all_labels).ravel()
        print(np.sum(all_labels))
        print(np.sum(all_preds))
        calculate_metrics(all_preds, all_labels)

    model.train()

    return running_loss / len(loader)

def main():

    train_dataset= MyOwnDataset("data/", "HIV_train.csv", False)
    test_dataset = MyOwnDataset("data/", "HIV_val.csv", test = True)
    
    train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle= True)
    test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle= True)

    model = GNN().to(device)
    weight = torch.tensor(config.loss_wieghts, dtype=torch.float32).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=config.LR,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma= config.gamma)

    model.train()
    early_stopping_counter = 0
    best_loss =1000

    for epoch in range(config.Num_epochs):

        if early_stopping_counter <= 10:

            #train_loss = train_one_epoch(epoch, train_loader, model, optimizer, loss_fn, scheduler)
            #print(f"Train loss after {epoch}: {train_loss}")

            if epoch % 5 == 0:

                val_loss = validation(epoch, test_loader, model, loss_fn)
                print(f"Validation loss after {epoch}: {val_loss}")

                if val_loss < best_loss:
                    best_loss = val_loss
                    early_stopping_counter = 0
                else:
                    early_stopping_counter +=1


        scheduler.step()

if __name__ == "__main__":
    main()