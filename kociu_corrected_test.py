"""
TEST CORRETTO - applica le due critiche piu' serie:
1. PROBE SET NEUTRALE: metriche calcolate su dati MAI usati per addestrare
   nessuna rete, identici per tutte, con scorciatoia BILANCIATA (neutra).
2. LEAVE-ONE-SEED-PAIR-OUT: quando testo il seed 7, escludo dal rilevatore
   ENTRAMBE le reti del seed 7 (baro + onesta gemella).
Se il rilevamento resta alto -> il segnale e' reale, non un artefatto.
Se crolla -> il 100% di prima era in parte spiegabile altrimenti.
"""
import torch, torch.nn as nn, numpy as np
from numpy.linalg import lstsq
torch.manual_seed(0); np.random.seed(0)

def make_data(n=600, shortcut_mode='present', seed=0):
    """shortcut_mode: 'present' (scorciatoia=etichetta), 'noise' (scorciatoia=rumore),
       'balanced' (scorciatoia neutra: meta' correlata meta' anti-correlata -> 0 informazione)"""
    rng=np.random.RandomState(seed); n_per=n//2
    a=rng.randn(n_per,2)+np.array([1.5,1.5]); b=rng.randn(n_per,2)+np.array([-1.5,-1.5])
    X2=np.concatenate([a,b]); y=np.concatenate([np.zeros(n_per),np.ones(n_per)])
    if shortcut_mode=='present':
        sc=y+rng.randn(n)*0.05
    elif shortcut_mode=='noise':
        sc=rng.randn(n)
    elif shortcut_mode=='balanced':
        # neutra: per meta' campioni segue y, per l'altra meta' segue 1-y -> correlazione netta ~0
        flip=rng.rand(n)<0.5
        sc=np.where(flip, y, 1-y)+rng.randn(n)*0.05
    X=np.column_stack([X2,sc]).astype(np.float32); X=(X-X.mean(0))/(X.std(0)+1e-9)
    return torch.tensor(X), torch.tensor(y.astype(np.float32)).unsqueeze(1)

class Net(nn.Module):
    def __init__(s,h=12):
        super().__init__(); s.fc1=nn.Linear(3,h); s.fc2=nn.Linear(h,h); s.fc3=nn.Linear(h,1); s.act=nn.Tanh()
    def hidden_rep(s,x): return s.act(s.fc2(s.act(s.fc1(x))))
    def forward(s,x): return torch.sigmoid(s.fc3(s.hidden_rep(x)))

def delta_phi_structured(H,labels):
    H=(H-H.mean(0))/(H.std(0)+1e-9); c0=H[labels==0]; c1=H[labels==1]
    m0,m1=c0.mean(0),c1.mean(0)
    within=(np.mean(np.linalg.norm(c0-m0,axis=1))+np.mean(np.linalg.norm(c1-m1,axis=1)))/2
    between=np.linalg.norm(m0-m1)
    return float(between/(within+between)) if (within+between)>1e-9 else 0.0
def psi_order(H):
    H=(H-H.mean(0))/(H.std(0)+1e-9); C=np.nan_to_num(np.corrcoef(H.T))
    eig=np.abs(np.linalg.eigvalsh(C)); s=eig.sum()
    if s<1e-12: return 0.0
    eig=eig/s; return float(1.0-(-np.sum(eig*np.log(eig+1e-12))/np.log(len(eig))))
def omega_stable(net,Xprobe,n_pert=50):
    # Ω mediato su molte perturbazioni (correzione: prima era 1 sola)
    vals=[]
    with torch.no_grad():
        H=net.hidden_rep(Xprobe)
        for _ in range(n_pert):
            Xp=Xprobe+torch.randn_like(Xprobe)*0.05; Hp=net.hidden_rep(Xp)
            dX=torch.norm(Xp-Xprobe,dim=1).numpy()+1e-9; dH=torch.norm(Hp-H,dim=1).numpy()
            if np.std(dH)>1e-9: vals.append(np.nan_to_num(np.corrcoef(dX,dH)[0,1]))
    return float(np.mean(vals)) if vals else 0.0

def train(net,X,Y,ep=1000):
    opt=torch.optim.Adam(net.parameters(),lr=0.01); lf=nn.BCELoss()
    for _ in range(ep): opt.zero_grad(); lf(net(X),Y).backward(); opt.step()
def acc(net,X,Y):
    with torch.no_grad(): return ((net(X)>0.5).float()==Y).float().mean().item()

# ---- PROBE SET NEUTRALE: unico, mai usato per addestrare, scorciatoia bilanciata ----
Xprobe,Yprobe=make_data(600,'balanced',seed=99999)
labels_probe=Yprobe.squeeze().numpy().astype(int)

N_PAIRS=25
profiles=[]; is_cheat=[]; seed_id=[]; test_gap=[]
print("Addestro %d coppie (baro + onesta gemella)..." % N_PAIRS)
for seed in range(N_PAIRS):
    Ytr=make_data(600,'present',seed+10)[1]
    # BARO: addestrata con scorciatoia presente
    Xc=make_data(600,'present',seed+10)[0]
    torch.manual_seed(seed); netC=Net(); train(netC,Xc,Ytr)
    # ONESTA: addestrata con scorciatoia=rumore
    Xo=make_data(600,'noise',seed+10)[0]
    torch.manual_seed(seed); netH=Net(); train(netH,Xo,Ytr)
    # METRICHE: calcolate SOLO sul probe neutrale, identico per entrambe
    for net,cheat in [(netC,1),(netH,0)]:
        with torch.no_grad(): Hp=net.hidden_rep(Xprobe).numpy()
        profiles.append([delta_phi_structured(Hp,labels_probe),psi_order(Hp),omega_stable(net,Xprobe)])
        is_cheat.append(cheat); seed_id.append(seed)
    # gap vero su test OOD (per riferimento)
    Xte,Yte=make_data(600,'noise',seed+500)
    test_gap.append((acc(netC,Xc,Ytr)-acc(netC,Xte,Yte), acc(netH,Xo,Ytr)-acc(netH,Xte,Yte)))

profiles=np.array(profiles); is_cheat=np.array(is_cheat); seed_id=np.array(seed_id)

# ---- LEAVE-ONE-SEED-PAIR-OUT ----
correct=0
for test_seed in range(N_PAIRS):
    train_mask = seed_id!=test_seed     # escludo ENTRAMBE le reti di questo seed
    test_mask  = seed_id==test_seed
    Xtr_=np.column_stack([np.ones(train_mask.sum()),profiles[train_mask]])
    beta,_,_,_=lstsq(Xtr_,is_cheat[train_mask],rcond=None)
    for idx in np.where(test_mask)[0]:
        pred=1 if (np.array([1,*profiles[idx]])@beta)>0.5 else 0
        if pred==is_cheat[idx]: correct+=1
detect=correct/len(profiles)

print("\n"+"="*64)
print("RISULTATO CON LE DUE CORREZIONI APPLICATE")
print("="*64)
print(f"Probe set: NEUTRALE (scorciatoia bilanciata, mai vista in training)")
print(f"Validazione: LEAVE-ONE-SEED-PAIR-OUT (esclude la gemella)")
print(f"\nRilevamento baro vs onesta: {detect*100:.1f}%")
print(f"\nProfili medi sul probe neutrale:")
print(f"  BARO:   dphi={profiles[is_cheat==1,0].mean():.3f}  psi={profiles[is_cheat==1,1].mean():.3f}  omega={profiles[is_cheat==1,2].mean():.3f}")
print(f"  ONESTA: dphi={profiles[is_cheat==0,0].mean():.3f}  psi={profiles[is_cheat==0,1].mean():.3f}  omega={profiles[is_cheat==0,2].mean():.3f}")
gaps=np.array(test_gap)
print(f"\nGap OOD reale (verifica): baro={gaps[:,0].mean()*100:.0f}%  onesta={gaps[:,1].mean()*100:.0f}%")
print("\n>> Se rilevamento ALTO: il segnale sopravvive alle correzioni -> reale.")
print(">> Se rilevamento ~50%: il 100% di prima era artefatto del probe non-neutrale.")
