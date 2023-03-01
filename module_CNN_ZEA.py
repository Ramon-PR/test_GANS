import torch
import numpy as np
# import matplotlib.pyplot as plt
import pymatreader as pymat
# from sklearn.metrics import accuracy_score

# criterion = torch.nn.CrossEntropyLoss()
# criterion = torch.nn.MSELoss() # sum( (xi-yi)**2 )/n
# optimizer = torch.optim.SGD(model.parameters(), lr=0.8)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)


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

        for col in range(0, img.shape[-1], self.stride):
            img2[:,:,col] = self.val
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

# --------------------------------------------------------------------
# MODEL
# --------------------------------------------------------------------
def dim_after_filter(dim_in, dim_kernel, pad, stripe):
    return int((dim_in + 2*pad - dim_kernel)/stripe) + 1

def block(chan_in, chan_out, kernel_size=3, pad1=1, str1=1, kernel_maxpool=2, str_maxpool=2):
    return torch.nn.Sequential(
        torch.nn.Conv2d(chan_in, chan_out, kernel_size, padding=pad1, stride=str1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_maxpool, stride=str_maxpool)
    )

def block2(c_in, c_out):
    return torch.nn.Sequential(
        torch.nn.Linear(c_in, c_out),
        torch.nn.ReLU()
    )

class CNN(torch.nn.Module):
  def __init__(self, n_channels=1, Hin=32, Win=32, Hout=32, Wout=32):
    super().__init__()

    chan_in = n_channels 

    n_filters1, n_filters2 = 5, 5
    kernel_size1, kernel_size2 = 3, 5
    pad1, pad2 = 1, 2
    str1, str2 = 1, 1

    kernel_maxpool, str_maxpool = 2, 2

    self.Hin = Hin
    self.Win = Win
    self.Hout = Hout
    self.Wout = Wout
    
    n_outputs = self.Hout* self.Wout

    self.conv1 = block(chan_in, n_filters1, kernel_size1, pad1, str1)
    H = dim_after_filter(Hin, kernel_size1, pad1, str1)
    W = dim_after_filter(Win, kernel_size1, pad1, str1)
    # if maxpool2d
    H, W = dim_after_filter(H, kernel_maxpool, 0, str_maxpool), dim_after_filter(W, kernel_maxpool, 0, str_maxpool) 

    self.conv2 = block(n_filters1, n_filters2, kernel_size2, pad2, str2)
    H = dim_after_filter(H, kernel_size2, pad2, str2)
    W = dim_after_filter(W, kernel_size2, pad2, str2)
    # if maxpool2d
    H, W = dim_after_filter(H, kernel_maxpool, 0, str_maxpool), dim_after_filter(W, kernel_maxpool, 0, str_maxpool)
    
    self.fc = torch.nn.Linear(n_filters2*H*W, n_outputs)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = x.view(x.shape[0], -1)
    x = self.fc(x)
    return x



def softmax(x):
    return torch.exp(x) / torch.exp(x).sum(axis=-1,keepdims=True)

# def evaluate(x):
#     model.eval()
#     y_pred = model(x)
#     y_probas = softmax(y_pred)
#     return torch.argmax(y_probas, axis=1)



def fit(model, dataloader, epochs=10, log_each=1, weight_decay=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    criterion = torch.nn.MSELoss() # sum( (xi-yi)**2 )/n
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    l, acc = [], []
    val_l, val_acc = [], []
    for e in range(1, epochs+1): 
        _l, _acc = [], []
        model.train()
        for x_b, y_b in dataloader['train']:
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
                y_b = y_b.view(y_b.shape[0],-1)
                y_pred = model(x_b)
                loss = criterion(y_pred, y_b)
                _l.append(loss.item())
                # y_probas = torch.argmax(softmax(y_pred), axis=1)            
                # _acc.append(accuracy_score(y_b.cpu().numpy(), y_probas.cpu().numpy()))
        val_l.append(np.mean(_l))
        # val_acc.append(np.mean(_acc))
        if not e % log_each:
            # print(f"Epoch {e}/{epochs} loss {l[-1]:.5f} acc {acc[-1]:.5f} val_loss {val_l[-1]:.5f} val_acc {val_acc[-1]:.5f}")
            print(f"Epoch {e}/{epochs} loss {l[-1]:.5f}  val_loss {val_l[-1]:.5f} ")
    # return {'epoch': list(range(1, epochs+1)), 'loss': l, 'acc': acc, 'val_loss': val_l, 'val_acc': val_acc}
    return {'epoch': list(range(1, epochs+1)), 'loss': l, 'val_loss': val_l}
 
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
