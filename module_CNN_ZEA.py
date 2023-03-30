import torch
import numpy as np
import pymatreader as pymat

device = "cuda" if torch.cuda.is_available() else "cpu"
# device="cpu"

def load_DB_ZEA(path, nx, nt):
    #Load data
    RIR=[]
    RIR = pymat.read_mat(path)["out"]["image"]
    T, M = RIR.shape

    # Divide RIR in small imagenes no-overlap
    # Num of imag in x and time
    tImags = T//nt
    xImags = M//nx
    nImags = (xImags)*(tImags)

    # Imags without modification
    Y = np.zeros([nImags,nt,nx])

    # Scaling
    # max_Image = RIR.max()
    # min_Image = RIR.min()

    for imx in range(xImags):
        for imt in range(tImags):
            im = imx*tImags + imt
            # Y[idx] is (nx,nt) if
            Y[im, :, :] = RIR[imt*nt : (imt+1)*nt, imx*nx : (imx+1)*nx]            
    return Y, RIR  


class down_samp_transf(torch.nn.Module):
    """Erase columns by giving value of val
    """
    def __init__(self, stride=2, val=0):
        super().__init__()
        self.stride = stride
        self.val = val

    def forward(self, img):
        # img2 = np.copy(img)
        img2 = torch.clone(img)

        for col in range(1, img.shape[-1], self.stride):
            img2[:,:,col] = self.val
        return img2

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(stride={self.stride})"


class down_samp2_transf(torch.nn.Module):
    """Select columns with information
    """
    def __init__(self, stride=2, val=0):
        super().__init__()
        self.stride = stride
        self.val = val

    def forward(self, img):
        # img2 = np.copy(img)
        img2 = torch.zeros(img.shape) + self.val

        for col in range(1, img.shape[-1], self.stride):
            img2[:,:,col] = img[:,:,col]
        return img2

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(stride={self.stride})"


class ZeaDataset(torch.utils.data.Dataset):
    # Constructor
    def __init__(self, X, target_transform=None, downsamp_transform=None):
    # def __init__(self, X, transform=None):
        # change to torch tensor, convert to float32 
        # and add a dimension at the beginning (1,H,W) 
        # to indicate the number of channels (batch,C,H,W)
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # self.X = torch.from_numpy(X).float().to(device).unsqueeze(1)
        self.X = torch.from_numpy(X).float().unsqueeze(1)
        self.target_transform = target_transform
        self.transform_inp = downsamp_transform
    
    # Return the number of data in our dataset
    def __len__(self):
        return len(self.X)
    
    # Return the element idx in the dataset
    def __getitem__(self, idx):
        image = self.X[idx]
        target= self.X[idx]
        if self.target_transform:
            target = self.target_transform(self.X[idx])
            image = target 

        if self.transform_inp:
            image = self.transform_inp(image)

        return image, target


def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(axis=-1,keepdims=True)

# def evaluate(x):
#     model.eval()
#     y_pred = model(x)
#     y_probas = softmax(y_pred)
#     return torch.argmax(y_probas, axis=1)


def fit(model, dataloader, optimizer, scheduler=None, epochs=100, log_each=10, weight_decay=0, early_stopping=0):
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    criterion = torch.nn.MSELoss() # sum( (xi-yi)**2 )/n
    l, acc, lr = [], [], []
    val_l, val_acc = [], []
    best_acc, step = 0, 0
    for e in range(1, epochs+1): 
        _l, _acc = [], []

# ------- Optimizer learning rate scheduler --------------------------
        for param_group in optimizer.param_groups:
            lr.append(param_group['lr'])
# --------------------------------------------------------------------

        model.train()
        for x_b, y_b in dataloader['train']:
            x_b, y_b = x_b.to(device), y_b.to(device)
            y_b = y_b.view(y_b.shape[0],-1)
            y_pred = model(x_b)
            loss = criterion(y_pred, y_b)
            _l.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # y_probas = torch.argmax(softmax(y_pred), axis=1)            
            # _acc.append(accuracy_score(y_b.cpu().numpy(), y_probas.cpu().detach().numpy()))
        l.append(np.mean(_l))
        # acc.append(np.mean(_acc))
        model.eval()
        _l, _acc = [], []
        with torch.no_grad():
            for x_b, y_b in dataloader['val']:
                x_b, y_b = x_b.to(device), y_b.to(device)
                y_b = y_b.view(y_b.shape[0],-1)
                y_pred = model(x_b)
                loss = criterion(y_pred, y_b)
                _l.append(loss.item())
                # y_probas = torch.argmax(softmax(y_pred), axis=1)            
                # _acc.append(accuracy_score(y_b.cpu().numpy(), y_probas.cpu().numpy()))
        val_l.append(np.mean(_l))
        # val_acc.append(np.mean(_acc))

# ---------------   Early stopping & best model ----------------------
        # guardar mejor modelo
        # if val_acc[-1] > best_acc:
        #     best_acc = val_acc[-1]
        #     torch.save(model.state_dict(), 'ckpt.pt')
        #     step = 0
        #     print(f"Mejor modelo guardado con acc {best_acc:.5f} en epoch {e}")
        # step += 1

# ------- Optimizer learning rate scheduler --------------------------
        if scheduler:
            scheduler.step()
# --------------------------------------------------------------------

    #     # parar
    #     if early_stopping and step > early_stopping:
    #         print(f"Entrenamiento detenido en epoch {e} por no mejorar en {early_stopping} epochs seguidas")
    #         break
    #     if not e % log_each:
    #         print(f"Epoch {e}/{epochs} loss {l[-1]:.5f} acc {acc[-1]:.5f} val_loss {val_l[-1]:.5f} val_acc {val_acc[-1]:.5f}")
    # # cargar mejor modelo
    # model.load_state_dict(torch.load('ckpt.pt'))
# --------------------------------------------------------------------

        if not e % log_each:
            # print(f"Epoch {e}/{epochs} loss {l[-1]:.5f} acc {acc[-1]:.5f} val_loss {val_l[-1]:.5f} val_acc {val_acc[-1]:.5f}")
            print(f"Epoch {e}/{epochs} loss {l[-1]:.5f}  val_loss {val_l[-1]:.5f} ")

    # return {'epoch': list(range(1, epochs+1)), 'loss': l, 'acc': acc, 'val_loss': val_l, 'val_acc': val_acc}
    # return {'epoch': list(range(1, epochs+1)), 'loss': l, 'val_loss': val_l}
    return {'epoch': list(range(1, len(l)+1)), 'loss': l, 'val_loss': val_l, 'lr': lr}





 
# !!! Validación cruzada !!!
# 
# from sklearn.model_selection import KFold
# FOLDS = 5
# kf = KFold(n_splits=FOLDS)
# X_train, X_test, y_train, y_test = X[:60000] / 255., X[60000:] / 255., Y[:60000].astype(np.int), Y[60000:].astype(np.int)
# X_train.shape, X_test.shape, kf.get_n_splits(X)
# train_accs, val_accs = [], []

# for k, (train_index, val_index) in enumerate(kf.split(X_test)):
#     print("Fold:", k+1)
#     X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
#     y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
#     dataset = {
#         'train': Dataset(X_train_fold, y_train_fold),
#         'val': Dataset(X_val_fold, y_val_fold)
#     }

#     dataloader = {
#         'train': torch.utils.data.DataLoader(dataset['train'], batch_size=100, shuffle=True),
#         'val': torch.utils.data.DataLoader(dataset['val'], batch_size=1000, shuffle=False)
#     }
    
#     model = build_model()
#     hist = fit(model, dataloader)
    
#     train_accs.append(hist['acc'][-1])   
#     val_accs.append(hist['val_acc'][-1])
