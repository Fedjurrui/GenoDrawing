
import torch,re,os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from tqdm import notebook
class snps_to_embd_dataset(Dataset):
    def __init__(self,folders,snps,encoder,decoder,device):
        self.images = []
        self.snps = []
        self.embd = []
        self.munq = []
        encoder.to(device)
        decoder.to(device)
        for dir in notebook.tqdm(folders):
            name = re.sub(".*\\\\","",dir)
            files = [os.path.join(dir,e) for e in os.listdir(dir)]
            embd = []
            images = [ToTensor()(Image.open(e)) for e in files]
            for i in images:
                if i.shape[1] != 300 or i.shape[2] != 300:
                    i = torchvision.transforms.Resize(size=(300,300))(i)
                embd.append(encoder(torch.reshape(i,shape=(1, 3, 300, 300)).to(device)).detach().cpu())
                del i
                torch.cuda.empty_cache()
            embd = torch.mean(torch.stack(embd),dim=0).to(device)
            decoded = decoder(embd)
            self.embd.append(embd[0].detach().cpu())
            self.images.append(decoded.detach().cpu()[0])
            self.snps.append(snps.loc[name,:].values/2)
            self.munq.append(name)

            del decoded
            del embd
            torch.cuda.empty_cache()
        del encoder
        del decoder
        torch.cuda.empty_cache()
    def __len__(self):
        return len(self.images)
    def __getitem__(self, item):
        return self.images[item],self.snps[item],self.embd[item],self.munq[item]