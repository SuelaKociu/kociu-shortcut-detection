"""
FRAMEWORK KOCIU - RILEVAMENTO DI EMERGENZA VERA
Bersaglio reale: distinguere una rete ONESTA (generalizza) da una che BARA
(impara una scorciatoia), guardando solo lo STATO INTERNO.
+ 5a metrica: flusso causale direzionale (Lambda).
Se il framework becca la rete-baro -> misura l'emergenza per davvero.
"""
import torch, torch.nn as nn, numpy as np
from scipy import stats
torch.manual_seed(0); np.random.seed(0)

# ---- DATI CON UNA SCORCIATOIA NASCOSTA ----
# Compito vero: classificare in base a 2 feature significative.
# Trappola: aggiungiamo una 3a feature che nei dati di TRAINING e' correlata
# con l'etichetta (scorciatoia), ma nei dati di TEST REALE NON lo e'.
# Rete onesta: usa le feature vere. Rete-baro: si aggrappa alla scorciatoia.
def make_data(n=600, shortcut_on=True, seed=0):
    rng=np.random.RandomState(seed); n_per=n//2
    # 2 feature vere (due blob separabili)
    a=rng.randn(n_per,2)+np.array([1.5,1.5]); b=rng.randn(n_per,2)+np.array([-1.5,-1.5])
    X2=np.concatenate([a,b]); y=np.concatenate([np.zeros(n_per),np.ones(n_per)])
    # 3a feature = scorciatoia
    if shortcut_on:
        shortcut=y+rng.randn(n)*0.05   # quasi perfettamente uguale all'etichetta
    else:
        shortcut=rng.randn(n)          # nel test reale e' solo rumore
    X=np.column_stack([X2,shortcut]).astype(np.float32)
    X=(X-X.mean(0))/(X.std(0)+1e-9)
    return torch.tensor(X), torch.tensor(y.astype(np.float32)).unsqueeze(1)

Xtr,Ytr=make_data(600,shortcut_on=True,seed=1)    # training: scorciatoia presente
Xte,Yte=make_data(600,shortcut_on=False,seed=2)   # test reale: scorciatoia sparita

class Net(nn.Module):
    def __init__(s,h=12):
        super().__init__(); s.fc1=nn.Linear(3,h); s.fc2=nn.Linear(h,h); s.fc3=nn.Linear(h,1); s.act=nn.Tanh()
    def hidden_rep(s,x): return s.act(s.fc2(s.act(s.fc1(x))))
    def forward(s,x): return torch.sigmoid(s.fc3(s.hidden_rep(x)))

# ---- LE 5 METRICHE ----
def delta_phi_structured(net,X,labels):
    net.eval()
    with torch.no_grad(): H=net.hidden_rep(X).numpy()
    H=(H-H.mean(0))/(H.std(0)+1e-9); c0=H[labels==0]; c1=H[labels==1]
    m0,m1=c0.mean(0),c1.mean(0)
    within=(np.mean(np.linalg.norm(c0-m0,axis=1))+np.mean(np.linalg.norm(c1-m1,axis=1)))/2
    between=np.linalg.norm(m0-m1)
    return float(between/(within+between)) if (within+between)>1e-9 else 0.0
def psi_order(net,X):
    net.eval()
    with torch.no_grad(): H=net.hidden_rep(X).numpy()
    H=(H-H.mean(0))/(H.std(0)+1e-9); C=np.nan_to_num(np.corrcoef(H.T))
    eig=np.abs(np.linalg.eigvalsh(C)); eig=eig/eig.sum()
    return float(1.0-(-np.sum(eig*np.log(eig+1e-12))/np.log(len(eig))))
def omega_continuity(net,X):
    net.eval()
    with torch.no_grad():
        H=net.hidden_rep(X); Xp=X+torch.randn_like(X)*0.05; Hp=net.hidden_rep(Xp)
        dX=torch.norm(Xp-X,dim=1).numpy()+1e-9; dH=torch.norm(Hp-H,dim=1).numpy()
        if np.std(dH)<1e-9: return 0.0
        return float(np.nan_to_num(np.corrcoef(dX,dH)[0,1]))
# 5a metrica LAMBDA: dipendenza causale dall'input-scorciatoia
# Quanto lo stato interno dipende dalla 3a feature (la scorciatoia)?
# Perturbo SOLO la feature-scorciatoia e misuro quanto cambia lo stato interno.
def lambda_shortcut_reliance(net,X,shortcut_idx=2):
    net.eval()
    with torch.no_grad():
        H=net.hidden_rep(X)
        Xp=X.clone(); Xp[:,shortcut_idx]+=1.0   # muovo solo la scorciatoia
        Hp=net.hidden_rep(Xp)
        reliance=torch.mean(torch.norm(Hp-H,dim=1)).item()
        # e quanto dipende dalle feature VERE (perturbo quelle)
        Xv=X.clone(); Xv[:,0]+=1.0; Xv[:,1]+=1.0
        Hv=net.hidden_rep(Xv)
        true_reliance=torch.mean(torch.norm(Hv-H,dim=1)).item()
    if (reliance+true_reliance)<1e-9: return 0.0
    return float(reliance/(reliance+true_reliance))  # frazione di dipendenza dalla scorciatoia

# ---- ALLENO DUE RETI: identiche fuori, diverse dentro ----
def train(net,X,Y,ep=1500):
    opt=torch.optim.Adam(net.parameters(),lr=0.01); lf=nn.BCELoss()
    for _ in range(ep): opt.zero_grad(); lf(net(X),Y).backward(); opt.step()

# rete A: allenata sui dati con scorciatoia (rischia di barare)
torch.manual_seed(1); netA=Net(); train(netA,Xtr,Ytr)
# rete B: allenata sugli stessi dati MA con la scorciatoia gia' resa inutile (rumore)
Xtr_clean,_=make_data(600,shortcut_on=False,seed=1)
torch.manual_seed(1); netB=Net(); train(netB,Xtr_clean,Ytr)

def acc(net,X,Y):
    with torch.no_grad(): return ((net(X)>0.5).float()==Y).float().mean().item()

labels_tr=Ytr.squeeze().numpy().astype(int)
print("="*66)
print("DUE RETI - come appaiono DALL'ESTERNO (accuratezza sul training):")
print("="*66)
print(f"  Rete A (esposta alla scorciatoia):  train acc = {acc(netA,Xtr,Ytr)*100:.1f}%")
print(f"  Rete B (senza scorciatoia utile):   train acc = {acc(netB,Xtr,Ytr)*100:.1f}%")
print("  --> Dall'esterno sul training sembrano simili.\n")
print("LA VERITA' - accuratezza sul TEST REALE (scorciatoia sparita):")
print(f"  Rete A: test acc = {acc(netA,Xte,Yte)*100:.1f}%   <-- se crolla, BARAVA")
print(f"  Rete B: test acc = {acc(netB,Xte,Yte)*100:.1f}%   <-- se regge, ONESTA")
print()
print("="*66)
print("IL FRAMEWORK GUARDA DENTRO - riesce a vederlo PRIMA del test reale?")
print("="*66)
for name,net in [("Rete A (sospetta)",netA),("Rete B (onesta)",netB)]:
    dphi=delta_phi_structured(net,Xtr,labels_tr)
    psi=psi_order(net,Xtr); ome=omega_continuity(net,Xtr)
    lam=lambda_shortcut_reliance(net,Xtr)
    print(f"\n{name}:")
    print(f"   Delta-phi (organizzazione):     {dphi:.3f}")
    print(f"   Psi (ordine):                   {psi:.3f}")
    print(f"   Omega (continuita'):            {ome:.3f}")
    print(f"   LAMBDA (dipendenza scorciatoia): {lam:.3f}  <-- LA METRICA CHIAVE")
print()
print(">> Se LAMBDA e' alto per A e basso per B, il framework ha RILEVATO il baro")
print("   guardando solo lo stato interno, senza bisogno del test reale.")
