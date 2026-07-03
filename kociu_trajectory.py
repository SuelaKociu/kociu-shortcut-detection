"""
ULTIMA STRADA: LA TRAIETTORIA TEMPORALE
Le firme statiche finali NON distinguono sparse/product da onesta.
Ipotesi: forse il MODO in cui imparano nel tempo e' diverso.
Salvo la rete a piu' checkpoint e guardo se la TRAIETTORIA distingue
cio' che lo stato finale non distingue.
Onesta': non so se funzionera'. Per questo vale la pena provarlo.
"""
import torch, torch.nn as nn, numpy as np
from numpy.linalg import lstsq
torch.manual_seed(0); np.random.seed(0)

def make_data(n=600, shortcut_type=None, seed=0):
    rng=np.random.RandomState(seed); n_per=n//2
    a=rng.randn(n_per,2)+np.array([1.5,1.5]); b=rng.randn(n_per,2)+np.array([-1.5,-1.5])
    X2=np.concatenate([a,b]); y=np.concatenate([np.zeros(n_per),np.ones(n_per)])
    cols=[X2]
    if shortcut_type=='sparse':
        s=rng.randn(n); idx=rng.rand(n)<0.3; s[idx]=y[idx]*10
        cols.append(s.reshape(-1,1)); cols.append(rng.randn(n,1))
    elif shortcut_type=='product':
        f1=rng.randn(n); f2=np.where(y>0.5,f1,-f1)+rng.randn(n)*0.1
        cols.append(f1.reshape(-1,1)); cols.append(f2.reshape(-1,1))
    elif shortcut_type=='noise':
        cols.append(rng.randn(n,1)); cols.append(rng.randn(n,1))
    X=np.column_stack(cols).astype(np.float32); X=(X-X.mean(0))/(X.std(0)+1e-9)
    return torch.tensor(X), torch.tensor(y.astype(np.float32)).unsqueeze(1)

IN_DIM=4
class Net(nn.Module):
    def __init__(s,h=16):
        super().__init__(); s.fc1=nn.Linear(IN_DIM,h); s.fc2=nn.Linear(h,h); s.fc3=nn.Linear(h,1); s.act=nn.Tanh()
    def hidden_rep(s,x): return s.act(s.fc2(s.act(s.fc1(x))))
    def forward(s,x): return torch.sigmoid(s.fc3(s.hidden_rep(x)))

def psi_order(H):
    H=(H-H.mean(0))/(H.std(0)+1e-9); C=np.nan_to_num(np.corrcoef(H.T))
    eig=np.abs(np.linalg.eigvalsh(C)); s=eig.sum()
    if s<1e-12: return 0.0
    eig=eig/s; return float(1.0-(-np.sum(eig*np.log(eig+1e-12))/np.log(len(eig))))

Xprobe,_=make_data(600,'noise',seed=99999)
CHECKPOINTS=[5,15,30,60,120,250,500,1000]

def trajectory_features(shortcut_type, seed):
    """Alleno salvando psi a ogni checkpoint -> la traiettoria di psi nel tempo."""
    Ytr=make_data(600,'sparse',seed+10)[1]  # etichette comuni
    Xtr=make_data(600,shortcut_type,seed+10)[0]
    torch.manual_seed(seed); net=Net()
    opt=torch.optim.Adam(net.parameters(),lr=0.01); lf=nn.BCELoss()
    psi_traj=[]; ep_done=0
    for target in CHECKPOINTS:
        while ep_done<target:
            opt.zero_grad(); lf(net(Xtr),Ytr).backward(); opt.step(); ep_done+=1
        with torch.no_grad(): Hp=net.hidden_rep(Xprobe).numpy()
        psi_traj.append(psi_order(Hp))
    psi_traj=np.array(psi_traj)
    # caratteristiche della TRAIETTORIA (non solo il valore finale):
    feats=[
        psi_traj[-1],                          # valore finale
        psi_traj.max()-psi_traj.min(),          # ampiezza del cambiamento
        np.max(np.abs(np.diff(psi_traj))),      # salto piu' brusco (transizione)
        np.argmax(np.abs(np.diff(psi_traj)))/len(psi_traj),  # QUANDO avviene il salto
        np.mean(psi_traj),                      # media
        np.std(psi_traj),                       # variabilita'
    ]
    return feats

print("Costruisco traiettorie temporali (sparse-baro vs onesta)...")
print("Confronto: la TRAIETTORIA distingue cio' che lo stato finale non distingueva?\n")

P=[]; lab=[]
for seed in range(18):
    P.append(trajectory_features('sparse',seed)); lab.append(1)   # baro sparse
    P.append(trajectory_features('noise',seed)); lab.append(0)    # onesta
P=np.array(P); lab=np.array(lab)

# leave-one-seed-pair-out
correct=0
seed_of=np.repeat(np.arange(18),2)
for ts in range(18):
    tr=seed_of!=ts
    Xm=np.column_stack([np.ones(tr.sum()),P[tr]])
    beta,_,_,_=lstsq(Xm,lab[tr],rcond=None)
    for idx in np.where(seed_of==ts)[0]:
        pred=1 if (np.array([1,*P[idx]])@beta)>0.5 else 0
        if pred==lab[idx]: correct+=1
detect=correct/len(P)

print("="*60)
print("RISULTATO: TRAIETTORIA TEMPORALE vs stato finale")
print("="*60)
print(f"Rilevamento sparse-baro con la TRAIETTORIA: {detect*100:.1f}%")
print(f"(con lo stato finale statico era: 0%)")
print(f"\nCaratteristiche medie della traiettoria di Psi:")
print(f"  {'':<22}{'sparse-baro':>14}{'onesta':>12}")
names=['valore finale','ampiezza','salto max','quando salto','media','std']
for i,nm in enumerate(names):
    print(f"  {nm:<22}{P[lab==1,i].mean():>14.3f}{P[lab==0,i].mean():>12.3f}")
print("\n>> Se rilevamento ALTO: la traiettoria vede cio' che lo stato finale")
print("   non vede -> il TEMPO contiene informazione che la fotografia perde.")
print(">> Se ~50%: nemmeno la traiettoria distingue -> sparse e' indistinguibile")
print("   dall'interno, un limite di principio.")
