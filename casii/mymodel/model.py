import torch
import torch.nn as nn
import torch.nn.functional as F

class attmil(nn.Module):

    def __init__(self, inputd=1024, hd1=512, hd2=256, k=None, r=None):
        super(attmil, self).__init__()

        self.hd1 = hd1
        self.hd2 = hd2
        self.feature_extractor = nn.Sequential(
            nn.Linear(1024, hd1),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(hd1, hd2),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(hd1, hd2),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(hd2, 1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=hd1, out_features=1)
        )


    def forward(self, x):
        x = self.feature_extractor(x) # mx512

        A_V = self.attention_V(x)  # mx256
        A_U = self.attention_U(x)  # mx256
        A = self.attention_weights(A_V * A_U) # element wise multiplication # mx1
        A = A.permute(0, 2, 1)  # 1xm
        A = F.softmax(A, dim=2)  # softmax over m

        M = torch.matmul(A, x)  # 1x512
        M = M.view(-1, self.hd1) # 512

        Y_prob = self.classifier(M)

        return Y_prob, A

class attmiltemp(nn.Module):

    def __init__(self, inputd=1024, hd1=512, hd2=256, k=None, r=None):
        super(attmiltemp, self).__init__()

        self.hd1 = hd1
        self.hd2 = hd2
        self.feature_extractor = nn.Sequential(
            nn.Linear(1024, hd1),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(hd1, hd2),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(hd1, hd2),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(hd2, 1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=hd1, out_features=1)
        )


    def forward(self, x):
        x = self.feature_extractor(x) # mx512

        A_V = self.attention_V(x)  # mx256
        A_U = self.attention_U(x)  # mx256
        A = self.attention_weights(A_V * A_U) # element wise multiplication # mx1
        A = A.permute(0, 2, 1)  # 1xm
        A = F.softmax(A/0.025, dim=2)  # softmax over m

        M = torch.matmul(A, x)  # 1x512
        M = M.view(-1, self.hd1) # 512

        Y_prob = self.classifier(M)

        return Y_prob, A

class attmilsc(nn.Module):

    def __init__(self, inputd=1024, hd1=512, hd2=256, k=None, r=None):
        super(attmilsc, self).__init__()

        self.hd1 = hd1
        self.hd2 = hd2
        self.feature_extractor = nn.Sequential(
            nn.Linear(1024, hd1),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(hd1, hd2),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(hd1, hd2),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(hd2, 1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=hd1, out_features=1)
        )


    def forward(self, x):
        x = self.feature_extractor(x) # mx512

        A_V = self.attention_V(x)  # mx256
        A_U = self.attention_U(x)  # mx256
        A = self.attention_weights(A_V * A_U) # element wise multiplication # mx1
        A = A.permute(0, 2, 1)  # 1xm
        A1 = F.softmax(A, dim=2)  # softmax over m

        M = torch.matmul(A1, x)  # 1x512
        M = M.view(-1, self.hd1) # 512

        Y_prob = self.classifier(M)

        return Y_prob, A1, A

class siiformer_NLattmil(nn.Module):

    def __init__(self, inputd=1024, hd=512, k=100, r=0.3):
        super(siiformer_NLattmil, self).__init__()

        self.hd = hd
        self.k = k
        self.r = r
        self.WQ = nn.Sequential(nn.Linear(inputd, hd), nn.Tanh())
        self.WK = nn.Sequential(nn.Linear(inputd, hd), nn.Tanh())
        self.attmil = attmil(inputd=1024, hd1=hd, hd2=hd//2,)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=hd, out_features=1)
        )


    def forward(self, x):
        # query, key = x
        query = self.WQ(x[0]) # mxhd 
        key = self.WK(x[1]) # nxhd

        q_norm = F.normalize(query, p=2, dim=2) # mxhd
        k_norm = F.normalize(key, p=2, dim=2) # nxhd
        # value = self.WV(query) # mxhd

        # Top-k keys
        k_norm = k_norm.transpose(2, 1) # hdxn
        A = torch.matmul(q_norm, k_norm) # mxn
        A, A_idx = torch.sort(A, descending=True) # mxn
        A_idx = A_idx.detach().cpu()
        A1 = torch.index_select(A, dim=-1, index=torch.arange(100))
        A1 = torch.mean(A1, dim=-1, keepdim=True) # mx1

        # Bottom r% queries
        
        test, sortidx = torch.sort(A1, dim=-2) # mx1
        test = test.detach().cpu()
        sal_query = torch.index_select(x[0], dim=1, index=sortidx[0, :int(self.r*A1.size(1)), 0])

        # AttMIL
        Y_prob, A1 = self.attmil(sal_query)

        return Y_prob, A1

class siiformer_ca(nn.Module):

    def __init__(self, inputd=1024, hd=512, k=100, r=0.3):
        super(siiformer_ca, self).__init__()

        self.hd = hd
        self.k = k
        self.r = r
        self.WQ = nn.Sequential(nn.Linear(inputd, hd), nn.Tanh())
        self.WK = nn.Sequential(nn.Linear(inputd, hd), nn.Tanh())
        self.WV = nn.Sequential(nn.Linear(inputd, hd), nn.ReLU())

        self.WA = nn.Linear(k, 1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=hd, out_features=1)
        )


    def forward(self, x):
        query, key = x
        query = self.WQ(query) # mxhd 
        key = self.WK(key) # mxhd
        
        q_norm = F.normalize(query, p=2, dim=2) # mxhd
        k_norm = F.normalize(key, p=2, dim=2) # nxhd
        # value = self.WV(query) # mxhd

        # Top-k keys
        k_norm = k_norm.transpose(2, 1) # hdxn
        A = torch.matmul(q_norm, k_norm) # mxn
        A, _ = torch.sort(A) # mxn
        A1 = A[:,:,-100:].clone() # mxk
        A1 = -self.WA(A1) # mx1

        # Bottom r% queries
        _, sortidx = torch.sort(A1, dim=-2) # mx1
        thre = float(A1[0, sortidx[0, -int(self.r*A1.size(1))], 0]) # find the bottom r% value as threshold
        A1 = F.threshold(A1, thre, 0) # mx1
        A1 = F.softmax(A1, dim=1) # mx1

        # Aggregation
        A1 = A1.permute(0, 2, 1) # 1xm
        z = torch.matmul(A1, query) # 1xhd
        z = z.view(-1, self.hd) # hd
        

        Y_prob = self.classifier(z)

        return Y_prob, A1

class siiformer_ca1(nn.Module):

    def __init__(self, inputd=1024, hd=512, k=100, r=0.3):
        super(siiformer_ca1, self).__init__()

        self.hd = hd
        self.k = k
        self.r = r
        self.WQ = nn.Sequential(nn.Linear(inputd, hd), nn.Tanh())
        self.WK = nn.Sequential(nn.Linear(inputd, hd), nn.Tanh())
        # self.WV = nn.Sequential(nn.Linear(inputd, hd), nn.ReLU())

        self.WA = nn.Linear(7933, 1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=hd, out_features=1)
        )


    def forward(self, x):
        query, key = x
        query = self.WQ(query) # mxhd 
        key = self.WK(key) # mxhd
        
        q_norm = F.normalize(query, p=2, dim=2) # mxhd
        k_norm = F.normalize(key, p=2, dim=2) # nxhd
        # value = self.WV(query) # mxhd

        # Top-k keys
        k_norm = k_norm.transpose(2, 1) # hdxn
        A = torch.matmul(q_norm, k_norm) # mxn
        A1 = self.WA(A) # mx1

        # Bottom r% queries
        _, sortidx = torch.sort(A1, dim=-2) # mx1
        thre = float(A1[0, sortidx[0, -int(self.r*A1.size(1))], 0]) # find the bottom r% value as threshold
        A1 = F.threshold(A1, thre, 0) # mx1
        A1 = F.softmax(A1, dim=1) # mx1

        # Aggregation
        A1 = A1.permute(0, 2, 1) # 1xm
        z = torch.matmul(A1, query) # 1xhd
        z = z.view(-1, self.hd) # hd
        

        Y_prob = self.classifier(z)

        return Y_prob, A1

class siiformer_casq(nn.Module):

    def __init__(self, inputd=1024, hd=512, k=100, r=0.3):
        super(siiformer_casq, self).__init__()

        self.hd = hd
        self.k = k
        self.r = r
        self.WQ = nn.Sequential(nn.Linear(inputd, hd), nn.Tanh())
        self.WK = nn.Sequential(nn.Linear(inputd, hd), nn.Tanh())
        self.WV = nn.Sequential(nn.Linear(inputd, hd), nn.ReLU())

        self.WA = nn.Linear(2*k, 1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=hd, out_features=1)
        )


    def forward(self, x):
        query, key = x
        query = self.WQ(query) # mxhd 
        key = self.WK(key) # mxhd
        
        q_norm = F.normalize(query, p=2, dim=2) # mxhd
        k_norm = F.normalize(key, p=2, dim=2) # nxhd
        # value = self.WV(query) # mxhd

        # Top-k keys
        k_norm = k_norm.transpose(2, 1) # hdxn
        A = torch.matmul(q_norm, k_norm) # mxn
        A, _ = torch.sort(A) # mxn
        A1 = A[:,:,-100:].clone() # mxk
        A2 = torch.square(A1)
        Acmb = torch.cat([A1, A2], -1)
        Acmb = -self.WA(Acmb) # mx1

        # Bottom r% queries
        _, sortidx = torch.sort(Acmb, dim=-2) # mx1
        thre = float(Acmb[0, sortidx[0, -int(self.r*Acmb.size(1))], 0]) # find the bottom r% value as threshold
        Acmb = F.threshold(Acmb, thre, 0) # mx1
        Acmb = F.softmax(Acmb, dim=1) # mx1

        # Aggregation
        Acmb = Acmb.permute(0, 2, 1) # 1xm
        z = torch.matmul(Acmb, query) # 1xhd
        z = z.view(-1, self.hd) # hd
        

        Y_prob = self.classifier(z)

        return Y_prob, Acmb