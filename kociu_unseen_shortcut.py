"""
IL TEST DECISIVO - SCORCIATOIA MAI VISTA
Addestro il rilevatore su UN tipo di scorciatoia, lo testo su tipi
COMPLETAMENTE DIVERSI mai visti durante la costruzione del rilevatore.
Se regge -> il framework cattura una firma GENERALE del "barare",
non un pattern specifico. Questa e' la prova grande.
"""
import torch, torch.nn as nn, numpy as np
from numpy.linalg import lstsq
torch.manual_seed(0); np.random.seed(0)

# ---- DIVERSI TIPI DI SCORCIATOIA (qualitativamente differenti) ----
def make_data(n=600, shortcut_type=None, seed=0):
    """shortcut_type: None=onesta(nessuna), oppure un tipo specifico di scorciatoia.
    Il compito vero e' sempre lo stesso (2 blob). Cambia COME si bara."""
    rng=np.random.RandomState(seed); n_per=n//2
    a=rng.randn(n_per,2)+np.array([1.5,1.5]); b=rng.randn(n_per,2)+np.array([-1.5,-1.5])
    X2=np.concatenate([a,b]); y=np.concatenate([np.zeros(n_per),np.ones(n_per)])
    cols=[X2]
    if shortcut_type=='linear':       # copia lineare dell'etichetta
        cols.append((y+rng.randn(n)*0.05).reshape(-1,1)); cols.append(rng.randn(n,1))
    elif shortcut_type=='threshold':  # scorciatoia a soglia (non lineare)
        s=np.where(y>0.5, rng.uniform(2,3,n), rng.uniform(-3,-2,n)); cols.append(s.reshape(-1,1)); cols.append(rng.randn(n,1))
    elif shortcut_type=='product':    # scorciatoia in 2 feature che vanno moltiplicate
        f1=rng.randn(n); f2=np.where(y>0.5,f1,-f1)+rng.randn(n)*0.1
        cols.append(f1.reshape(-1,1)); cols.append(f2.reshape(-1,1))
    elif shortcut_type=='sparse':     # scorciatoia attiva solo su pochi campioni (rara ma fortissima)
        s=rng.randn(n); idx=rng.rand(n)<0.3; s[idx]=y[idx]*10
        cols.append(s.reshape(-1,1)); cols.append(rng.randn(n,1))
    elif shortcut_type=='noise':      # onesta: 2 colonne di puro rumore
        cols.append(rng.randn(n,1)); cols.append(rng.randn(n,1))
    X=np.column_stack(cols).astype(np.float32); X=(X-X.mean(0))/(X.std(0)+1e-9)
    return torch.tensor(X), torch.tensor(y.astype(np.float32)).unsqueeze(1)

IN_DIM=4  # 2 vere + 2 colonne scorciatoia/rumore

class Net(nn.Module):
    def __init__(s,h=16):
        super().__init__(); s.fc1=nn.Linear(IN_DIM,h); s.fc2=nn.Linear(h,h); s.fc3=nn.Linear(h,1); s.act=nn.Tanh()
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
def omega_stable(net,Xp,n_pert=40):
    vals=[]
    with torch.no_grad():
        H=net.hidden_rep(Xp)
        for _ in range(n_pert):
            Xq=Xp+torch.randn_like(Xp)*0.05; Hq=net.hidden_rep(Xq)
            dX=torch.norm(Xq-Xp,dim=1).numpy()+1e-9; dH=torch.norm(Hq-H,dim=1).numpy()
            if np.std(dH)>1e-9: vals.append(np.nan_to_num(np.corrcoef(dX,dH)[0,1]))
    return float(np.mean(vals)) if vals else 0.0

def train(net,X,Y,ep=1000):
    opt=torch.optim.Adam(net.parameters(),lr=0.01); lf=nn.BCELoss()
    for _ in range(ep): opt.zero_grad(); lf(net(X),Y).backward(); opt.step()

# probe neutrale unico
Xprobe,Yprobe=make_data(600,'noise',seed=99999)
labels_probe=Yprobe.squeeze().numpy().astype(int)

def build_profiles(shortcut_types, n_per_type=12):
    P=[]; cheat=[]
    for st in shortcut_types:
        for seed in range(n_per_type):
            Ytr=make_data(600,'linear',seed+10)[1]  # etichette comuni
            is_honest = (st=='noise')
            Xtr=make_data(600,st,seed+10)[0]
            torch.manual_seed(seed); net=Net(); train(net,Xtr,Ytr)
            with torch.no_grad(): Hp=net.hidden_rep(Xprobe).numpy()
            P.append([delta_phi_structured(Hp,labels_probe),psi_order(Hp),omega_stable(net,Xprobe)])
            cheat.append(0 if is_honest else 1)
    return np.array(P), np.array(cheat)

print("Costruisco il rilevatore su scorciatoia 'linear' + reti oneste...")
P_train, c_train = build_profiles(['linear','noise'], n_per_type=15)
# alleno il classificatore (regressione) SOLO su linear+oneste
Xm=np.column_stack([np.ones(len(P_train)),P_train])
beta,_,_,_=lstsq(Xm,c_train,rcond=None)

print("Testo su scorciatoie MAI VISTE: threshold, product, sparse\n")
print("="*60)
print("TEST DI GENERALIZZAZIONE A SCORCIATOIE NUOVE")
print("="*60)
print(f"Rilevatore addestrato SOLO su: 'linear' + oneste")
print(f"{'Tipo scorciatoia (mai vista)':<32}{'rilevata come baro'}")
print("-"*60)
for st in ['threshold','product','sparse']:
    P_test,_=build_profiles([st], n_per_type=15)
    preds=[1 if (np.array([1,*p])@beta)>0.5 else 0 for p in P_test]
    rate=np.mean(preds)
    print(f"{st:<32}{rate*100:>5.0f}%")
# controllo: reti oneste nuove devono NON essere rilevate come baro
P_honest,_=build_profiles(['noise'], n_per_type=15)
preds_h=[1 if (np.array([1,*p])@beta)>0.5 else 0 for p in P_honest]
print(f"{'(controllo) oneste nuove':<32}{np.mean(preds_h)*100:>5.0f}%  <- deve essere BASSO")
print("\n>> Alto sui tre tipi nuovi + basso sul controllo = il framework")
print("   cattura una firma GENERALE del barare, non un pattern memorizzato.")
