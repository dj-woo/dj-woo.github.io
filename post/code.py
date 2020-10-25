import torch
from torch.utils.data import DataLoader, Dataset

class TempDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        in_data = torch.FloatTensor(self.x[idx])
        out_data = torch.FloatTensor(self.y[idx])
        return (in_data,out_data)

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N,D_in)
y = torch.randn(N,D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

lr = 0.001
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(),lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

x_dataset = TempDataset(x,y)
dataloader = DataLoader(dataset=x_dataset,batch_size=4,shuffle=True)

for t in range(10):
    loss_epoch = 0
    for i_batch, (src,trg) in enumerate(dataloader):
        y_pre = model(src)
        loss = loss_fn(y_pre,trg)
        loss.backward()
        with torch.no_grad():
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        loss_epoch += loss

    print(t,loss_epoch.item())

