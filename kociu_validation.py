"""
VALIDAZIONE RIGOROSA - Delta-phi strutturato
Test su MOLTE reti (semi diversi), MOLTI livelli di addestramento, PIU' compiti.
Domanda: la correlazione r~0.94 regge, o era fortuna di un singolo run?
Nessun target. Controlli inclusi. Tutto ripetibile.
"""
import torch, torch.nn as nn, numpy as np
from scipy import stats

def make_spirals(n=400, seed=0):
    rng=np.random.RandomState(seed); n_per=n//2
    theta=np.sqrt(rng.rand(n_per))*3*np.pi
    xa=np.stack([theta*np.cos(theta),theta*np.sin(theta)],1)+rng.randn(n_per,2)*0.3
    xb=np.stack([-theta*np.cos(theta),-theta*np.sin(theta)],1)+rng.randn(n_per,2)*0.3
    X=np.concatenate([xa,xb]).astype(np.float32); X=(X-X.mean(0))/X.std(0)
    Y=np.concatenate([np.zeros(n_per),np.ones(n_per)]).astype(np.float32)
    return torch.tensor(X),torch.tensor(Y).unsqueeze(1)

def make_circles(n=400, seed=0):
    rng=np.random.RandomState(seed); n_per=n//2
    t=rng.rand(n_per)*2*np.pi
    inner=np.stack([np.cos(t),np.sin(t)],1)*1.0+rng.randn(n_per,2)*0.1
    outer=np.stack([np.cos(t),np.sin(t)],1)*2.5+rng.randn(n_per,2)*0.1
    X=np.concatenate([inner,outer]).astype(np.float32); X=(X-X.mean(0))/X.std(0)
    Y=np.concatenate([np.zeros(n_per),np.ones(n_per)]).astype(np.float32)
    return torch.tensor(X),torch.tensor(Y).unsqueeze(1)

class Net(nn.Module):
    def __init__(s,h=12):
        super().__init__(); s.fc1=nn.Linear(2,h); s.fc2=nn.Linear(h,h); s.fc3=nn.Linear(h,1); s.act=nn.Tanh()
    def hidden_rep(s,x): return s.act(s.fc2(s.act(s.fc1(x))))
    def forward(s,x): return torch.sigmoid(s.fc3(s.hidden_rep(x)))

def delta_phi_structured(net,X,labels):
    net.eval()
    with torch.no_grad(): H=net.hidden_rep(X).numpy()
    H=(H-H.mean(0))/(H.std(0)+1e-9)
    c0=H[labels==0]; c1=H[labels==1]; m0,m1=c0.mean(0),c1.mean(0)
    within=(np.mean(np.linalg.norm(c0-m0,axis=1))+np.mean(np.linalg.norm(c1-m1,axis=1)))/2
    between=np.linalg.norm(m0-m1)
    return float(between/(within+between)) if (within+between)>1e-9 else 0.0

def run_validation(task_fn, task_name, n_nets=30):
    print(f"\n=== VALIDAZIONE su {task_name} ({n_nets} reti, addestramento variabile) ===")
    accs, dphis = [], []
    for seed in range(n_nets):
        torch.manual_seed(seed)
        X,Y=task_fn(seed=seed); labels=Y.squeeze().numpy().astype(int)
        net=Net(); opt=torch.optim.Adam(net.parameters(),lr=0.01); lf=nn.BCELoss()
        # numero di epoche VARIABILE: copre tutto lo spettro da non-allenata a ben-allenata
        n_epochs = int(np.random.RandomState(seed).choice([0,20,50,100,200,500,1500,3000]))
        for _ in range(n_epochs):
            opt.zero_grad(); lf(net(X),Y).backward(); opt.step()
        with torch.no_grad(): acc=(((net(X)>0.5).float())==Y).float().mean().item()
        accs.append(acc); dphis.append(delta_phi_structured(net,X,labels))
    accs,dphis=np.array(accs),np.array(dphis)
    r,p=stats.pearsonr(accs,dphis)
    rho,p_s=stats.spearmanr(accs,dphis)
    print(f"  Pearson r  = {r:.3f}  (p={p:.2e})")
    print(f"  Spearman ρ = {rho:.3f}  (p={p_s:.2e})")
    print(f"  Range accuratezza osservato: {accs.min()*100:.0f}% - {accs.max()*100:.0f}%")
    print(f"  Range delta-phi osservato:   {dphis.min():.3f} - {dphis.max():.3f}")
    return r, p, accs, dphis

r1,p1,a1,d1 = run_validation(make_spirals,"SPIRALI",30)
r2,p2,a2,d2 = run_validation(make_circles,"CERCHI CONCENTRICI",30)

# Controllo cruciale: la metrica correla con accuratezza VERA, o con etichette a caso?
print("\n=== CONTROLLO: etichette RANDOMIZZATE (deve far CROLLARE r) ===")
accs_shuf, dphis_shuf = [], []
for seed in range(30):
    torch.manual_seed(seed)
    X,Y=make_spirals(seed=seed)
    labels_true=Y.squeeze().numpy().astype(int)
    labels_fake=np.random.RandomState(seed+999).permutation(labels_true)  # etichette finte
    net=Net(); opt=torch.optim.Adam(net.parameters(),lr=0.01); lf=nn.BCELoss()
    n_epochs=int(np.random.RandomState(seed).choice([0,50,200,1500,3000]))
    for _ in range(n_epochs): opt.zero_grad(); lf(net(X),Y).backward(); opt.step()
    with torch.no_grad(): acc=(((net(X)>0.5).float())==Y).float().mean().item()
    accs_shuf.append(acc)
    dphis_shuf.append(delta_phi_structured(net,X,labels_fake))  # metrica su etichette FINTE
r_shuf,p_shuf=stats.pearsonr(accs_shuf,dphis_shuf)
print(f"  Con etichette finte: Pearson r = {r_shuf:.3f} (p={p_shuf:.2e})")
print(f"  >> Atteso: r CROLLA verso 0. Se cosi', la metrica misura organizzazione REALE col compito.")

print("\n=== VERDETTO ===")
print(f"Spirali:  r={r1:.3f} (p={p1:.1e})")
print(f"Cerchi:   r={r2:.3f} (p={p2:.1e})")
print(f"Controllo etichette-finte: r={r_shuf:.3f}")
print(f"Media |r| sui compiti reali: {(abs(r1)+abs(r2))/2:.3f}")
