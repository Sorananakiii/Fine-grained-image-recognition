

class AdaptiveAM(nn.Module):
    ''' version 1 : of AAM loss '''
    def __init__(self, in_feat, out_feat, s=30.0, m1=0.50):
        super(AdaptiveAM, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.s = s
        self.m1 = m1
        self.weight = Parameter(torch.FloatTensor(out_feat, in_feat))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label, epoch):
        #---------------------------- Margin Additional -----------------------------
        m1 = self.m1

        cos_m, sin_m = math.cos(m1), math.sin(m1)
        th = math.cos(math.pi - m1)
        mm = math.sin(math.pi - m1) * m1
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        feature = F.linear(input, self.weight)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        angle = torch.acos(cosine)

        one_hot = torch.zeros(cosine.size()).to(device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        addin = (onehot * m1 * angle) + ((1.0 - one_hot) * angle) 
        # --------------------------- fixed to Q1 ---------------------------
        # phi = torch.where(cosine > th, phi, torch.zeros(1).to(device))
        output = torch.norm(input, dim=0) * torch.cos(addin)

        return feature, output