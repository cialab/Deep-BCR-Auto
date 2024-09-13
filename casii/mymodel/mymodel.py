import torch
import torch.nn as nn
import torch.nn.functional as F




class CASiiHead(nn.Module):
    def __init__(self, inputd=1024, hd=512, k=10, A_dim=3200, tau=1):
        super(CASiiHead, self).__init__()

        self.hd = hd
        self.k = k
        self.tau = tau
        self.WQ = nn.Sequential(nn.Linear(inputd, hd), nn.Tanh())
        self.WK = nn.Sequential(nn.Linear(inputd, hd), nn.Tanh())
        self.WV = nn.Sequential(nn.Linear(inputd, hd), nn.ReLU())

        self.WA = nn.Linear(A_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=hd, out_features=1)
        )

    def metafusion(self, A):
        A = self.WA(A).squeeze(-1) # m
        topq, _ = torch.topk(A, int(self.k), dim=-1) # top-k query logits
        botq, _ = torch.topk(A, int(self.k), dim=-1, largest=False) # bot-k quesry logits
        A = F.softmax(A/self.tau, dim=1).unsqueeze(2) # mx1
        return A, topq, botq

    def forward(self, x):
        query, key = x
        value = self.WV(query) # mxhd
        query = self.WQ(query) # mxhd 
        key = self.WK(key) # mxhd
        
        q_norm = F.normalize(query, p=2, dim=2) # mxhd
        k_norm = F.normalize(key, p=2, dim=2) # nxhd

        # Top-k keys
        k_norm = k_norm.transpose(2, 1) # hdxn
        A = torch.matmul(q_norm, k_norm) # mxn
        
        # metafusion
        A, topq, botq = self.metafusion(A) # mx1

        # Aggregation
        A = A.permute(0, 2, 1) # 1xm
        z = torch.matmul(A, value) # 1xhd
        z = z.view(-1, self.hd) # hd
        

        Y_prob = self.classifier(z)

        return Y_prob, A


class CASii_MB(nn.Module):
    def __init__(self, inputd=1024, hd=512, n_head=2, A_dims=[4000, 4000], k=1, tau=1):
        super(CASii_MB, self).__init__()

        self.hd = hd
        self.k = k
        self.tau = tau
        self.n_head = n_head

        self.heads = nn.ModuleList([CASiiHead(inputd=inputd, hd=hd, k=k, A_dim=d) for d in A_dims])
        
    def forward(self, x):
        logits = torch.empty(1, self.n_head).float().cuda()
        As = [None] * self.n_head
        for c in range(self.n_head):
            logits[0, c], As[c] = self.heads[c]([x[0], x[c+1]])
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        # Y_prob = F.softmax(logits, dim = 1)

        return Y_hat, logits, As

