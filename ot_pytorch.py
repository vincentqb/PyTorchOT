import torch


def sink(M, reg, numItermax=1000, stopThr=1e-9):
    # we assume that no distances are null except those of the diagonal of distances

    a = torch.ones((M.size()[0],), requires_grad=True, device=M.device) / M.size()[0]
    b = torch.ones((M.size()[1],), requires_grad=True, device=M.device) / M.size()[1]

    # init data
    Nini = len(a)
    Nfin = len(b)

    u = torch.ones(Nini, requires_grad=True, device=M.device) / Nini
    v = torch.ones(Nfin, requires_grad=True, device=M.device) / Nfin

    K = torch.exp(-M / reg)

    Kp = (1 / a).view(-1, 1) * K
    cpt = 0
    err = 1
    while err > stopThr and cpt < numItermax:
        KtransposeU = K.t().matmul(u)
        v = torch.div(b, KtransposeU)
        u = 1.0 / Kp.matmul(v)

        if cpt % 10 == 0:
            # we can speed up the process by checking for the error only all the 10th iterations
            transp = u.view(-1, 1) * (K * v)
            err = (torch.sum(transp, dim=0) - b).norm(2).pow(2).detach().cpu().item()

        cpt += 1

    return torch.sum(u.view((-1, 1)) * K * v.view((1, -1)) * M)


def sink_stabilized(M, reg, numItermax=1000, tau=1e2, stopThr=1e-9, warmstart=None, print_period=20):
    a = torch.ones((M.size()[0],), requires_grad=True, device=M.device) / M.size()[0]
    b = torch.ones((M.size()[1],), requires_grad=True, device=M.device) / M.size()[1]

    # init data
    na = len(a)
    nb = len(b)

    cpt = 0
    # we assume that no distances are null except those of the diagonal of distances
    if warmstart is None:
        alpha = torch.zeros(na, requires_grad=True, device=M.device)
        beta = torch.zeros(nb, requires_grad=True, device=M.device)
    else:
        alpha, beta = warmstart

    u = torch.ones(na, requires_grad=True, device=M.device) / na
    v = torch.ones(nb, requires_grad=True, device=M.device) / nb

    def get_K(alpha, beta):
        return torch.exp(-(M - alpha.view((na, 1)) - beta.view((1, nb))) / reg)

    def get_Gamma(alpha, beta, u, v):
        return torch.exp(
            -(M - alpha.view((na, 1)) - beta.view((1, nb))) / reg
            + torch.log(u.view((na, 1)))
            + torch.log(v.view((1, nb)))
        )

    K = get_K(alpha, beta)
    transp = K
    loop = 1
    cpt = 0
    err = 1
    while loop:
        # sinkhorn update
        v = torch.div(b, (K.t().matmul(u) + 1e-16))
        u = torch.div(a, (K.matmul(v) + 1e-16))

        # remove numerical problems and store them in K
        if torch.max(torch.abs(u)).detach().cpu().item() > tau or torch.max(torch.abs(v)).detach().cpu().item() > tau:
            alpha, beta = alpha + reg * torch.log(u), beta + reg * torch.log(v)

            u = torch.ones(na, requires_grad=True, device=M.device) / na
            v = torch.ones(nb, requires_grad=True, device=M.device) / nb

            K = get_K(alpha, beta)

        if cpt % print_period == 0:
            transp = get_Gamma(alpha, beta, u, v)
            err = (torch.sum(transp, dim=0) - b).norm(2).pow(2).detach().cpu().item()

        if err <= stopThr:
            loop = False

        if cpt >= numItermax:
            loop = False

        cpt += 1

    return torch.sum(get_Gamma(alpha, beta, u, v) * M)


def pairwise_distances(x, y, method="l1"):
    n = x.size()[0]
    m = y.size()[0]
    d = x.size()[1]

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    if method == "l1":
        dist = torch.abs(x - y).sum(2)
    else:
        dist = torch.pow(x - y, 2).sum(2)

    return dist.float()
