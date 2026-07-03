"""
DARE AL FRAMEWORK UN SECONDO SENSO
Le metriche attuali vedono solo la COMPRESSIONE interna -> cieche a scorciatoie
sparse/moltiplicative. Aggiungo una metrica che misura la DIPENDENZA CAUSALE
dall'input: quanto la rete si appoggia a feature che, perturbate, cambiano molto
l'output IN MODO CONCENTRATO su poche feature.
Test: addestro su 'linear', testo su threshold/product/sparse mai viste.
"""
import torch, torch.nn as nn, numpy as np
from numpy.linalg import lstsq
torch.manual_seed(0); np.random.seed(0)

def make_data(n=600, shortcut_type=None, seed=0):
    rng=np.random.RandomState(seed); n_per=n//2
    a=rng.randn(n_per,2)+np.array([1.5,1.5]); b=rng.randn(n_per,2)+np.array([-1.5,-1.5])
    X2=np.concatenate([a,b]); y=np.concatenate([np.zeros(n_per),np.ones(n_per)])
    cols=[X2]
    if shortcut_type=='linear':
        cols.append((y+rng.randn(n)*0.05).reshape(-1,1)); cols.append(rng.randn(n,1))
    elif shortcut_type=='threshold':
        s=np.where(y>0.5, rng.uniform(2,3,n), rng.uniform(-3,-2,n)); cols.append(s.reshape(-1,1)); cols.append(rng.randn(n,1))
    elif shortcut_type=='product':
        f1=rng.randn(n); f2=np.where(y>0.5,f1,-f1)+rng.randn(n)*0.1
        cols.append(f1.reshape(-1,1)); cols.append(f2.reshape(-1,1))
    elif shortcut_type=='sparse':
        s=rng.randn(n); idx=rng.rand(n)<0.3; s[idx]=y[idx]*10
        cols.append(s.reshape(-1,1)); cols.append(rng.randn(n,1))
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

# ---- SECONDO SENSO: dipendenza causale CONCENTRATA dall'input ----
# Perturbo OGNI feature di input separatamente, misuro quanto cambia l'output.
# Una rete ONESTA usa le 2 feature vere in modo distribuito.
# Una rete BARO concentra la dipendenza su poche feature (incluse le spurie).
# Metrica = concentrazione della dipendenza (Gini-like): alta = si appoggia a poche feature.
def causal_input_concentration(net, Xprobe, eps=0.5):
    net.eval()
    with torch.no_grad():
        base=net(Xprobe)
        effects=[]
        for j in range(Xprobe.shape[1]):
            Xp=Xprobe.clone(); Xp[:,j]+=eps
            out=net(Xp)
            effects.append(torch.mean(torch.abs(out-base)).item())
        effects=np.array(effects)
    if effects.sum()<1e-9: return 0.0
    p=effects/effects.sum()
    # concentrazione = 1 - entropia normalizzata: alta se dipende da POCHE feature
    ent=-np.sum(p*np.log(p+1e-12))/np.log(len(p))
    return float(1.0-ent)

def train(net,X,Y,ep=1000):
    opt=torch.optim.Adam(net.parameters(),lr=0.01); lf=nn.BCELoss()
    for _ in range(ep): opt.zero_grad(); lf(net(X),Y).backward(); opt.step()

Xprobe,Yprobe=make_data(600,'noise',seed=99999)
labels_probe=Yprobe.squeeze().numpy().astype(int)

def build_profiles(shortcut_types, n_per_type=15):
    P=[]; cheat=[]
    for st in shortcut_types:
        for seed in range(n_per_type):
            Ytr=make_data(600,'linear',seed+10)[1]
            Xtr=make_data(600,st,seed+10)[0]
            torch.manual_seed(seed); net=Net(); train(net,Xtr,Ytr)
            with torch.no_grad(): Hp=net.hidden_rep(Xprobe).numpy()
            # ora 3 metriche: le 2 vecchie + il SECONDO SENSO causale
            P.append([delta_phi_structured(Hp,labels_probe), psi_order(Hp),
                      causal_input_concentration(net,Xprobe)])
            cheat.append(0 if st=='noise' else 1)
    return np.array(P), np.array(cheat)

print("Costruisco rilevatore su 'linear'+oneste, CON il secondo senso causale...")
P_train,c_train=build_profiles(['linear','noise'])
Xm=np.column_stack([np.ones(len(P_train)),P_train])
beta,_,_,_=lstsq(Xm,c_train,rcond=None)

print("\n"+"="*62)
print("TEST SCORCIATOIE MAI VISTE - CON SECONDO SENSO (dip. causale)")
print("="*62)
print(f"{'Tipo (mai visto)':<28}{'rilevata':<12}{'(prima era)'}")
print("-"*62)
prev={'threshold':'100%','product':'47%','sparse':'0%'}
for st in ['threshold','product','sparse']:
    P_test,_=build_profiles([st])
    rate=np.mean([1 if (np.array([1,*p])@beta)>0.5 else 0 for p in P_test])
    print(f"{st:<28}{rate*100:>4.0f}%{'':<7}{prev[st]}")
P_h,_=build_profiles(['noise'])
rate_h=np.mean([1 if (np.array([1,*p])@beta)>0.5 else 0 for p in P_h])
print(f"{'(controllo) oneste':<28}{rate_h*100:>4.0f}%{'':<7}0% (deve restare basso)")

# mostro i valori del secondo senso per capire se separa
print("\nValore medio del SECONDO SENSO (concentrazione causale):")
for st in ['linear','threshold','product','sparse','noise']:
    P_st,_=build_profiles([st], n_per_type=10)
    print(f"  {st:<12}: {P_st[:,2].mean():.3f}")
