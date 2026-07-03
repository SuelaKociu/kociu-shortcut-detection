"""
STRESS TEST DEL FRAMEWORK KOCIU
Proviamo a ROMPERLO: scale diverse, tipi di baro diversi, architetture diverse.
Onesta' totale: riportiamo dove regge E dove fallisce.
"""
import torch, torch.nn as nn, numpy as np
from scipy import stats
from numpy.linalg import lstsq
torch.manual_seed(0); np.random.seed(0)

# ---- METRICHE (identiche a prima) ----
def delta_phi_structured(H_np, labels):
    H=(H_np-H_np.mean(0))/(H_np.std(0)+1e-9); c0=H[labels==0]; c1=H[labels==1]
    if len(c0)<2 or len(c1)<2: return 0.0
    m0,m1=c0.mean(0),c1.mean(0)
    within=(np.mean(np.linalg.norm(c0-m0,axis=1))+np.mean(np.linalg.norm(c1-m1,axis=1)))/2
    between=np.linalg.norm(m0-m1)
    return float(between/(within+between)) if (within+between)>1e-9 else 0.0
def psi_order(H_np):
    H=(H_np-H_np.mean(0))/(H_np.std(0)+1e-9); C=np.nan_to_num(np.corrcoef(H.T))
    eig=np.abs(np.linalg.eigvalsh(C)); s=eig.sum()
    if s<1e-12: return 0.0
    eig=eig/s; return float(1.0-(-np.sum(eig*np.log(eig+1e-12))/np.log(len(eig))))
def omega_continuity(net,X):
    with torch.no_grad():
        H=net.hidden_rep(X); Xp=X+torch.randn_like(X)*0.05; Hp=net.hidden_rep(Xp)
        dX=torch.norm(Xp-X,dim=1).numpy()+1e-9; dH=torch.norm(Hp-H,dim=1).numpy()
        if np.std(dH)<1e-9: return 0.0
        return float(np.nan_to_num(np.corrcoef(dX,dH)[0,1]))

def make_data(n=600, shortcut_on=True, seed=0, n_shortcut=1):
    rng=np.random.RandomState(seed); n_per=n//2
    a=rng.randn(n_per,2)+np.array([1.5,1.5]); b=rng.randn(n_per,2)+np.array([-1.5,-1.5])
    X2=np.concatenate([a,b]); y=np.concatenate([np.zeros(n_per),np.ones(n_per)])
    cols=[X2]
    for k in range(n_shortcut):
        cols.append((y+rng.randn(n)*0.05).reshape(-1,1) if shortcut_on else rng.randn(n).reshape(-1,1))
    X=np.column_stack(cols).astype(np.float32); X=(X-X.mean(0))/(X.std(0)+1e-9)
    return torch.tensor(X), torch.tensor(y.astype(np.float32)).unsqueeze(1)

def build_net(in_dim, hidden, depth=2, act='tanh'):
    A=nn.Tanh if act=='tanh' else nn.ReLU
    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            layers=[nn.Linear(in_dim,hidden),A()]
            for _ in range(depth-1): layers+=[nn.Linear(hidden,hidden),A()]
            s.body=nn.Sequential(*layers); s.head=nn.Linear(hidden,1)
        def hidden_rep(s,x): return s.body(x)
        def forward(s,x): return torch.sigmoid(s.head(s.hidden_rep(x)))
    return Net()

def train(net,X,Y,ep=1000):
    opt=torch.optim.Adam(net.parameters(),lr=0.01); lf=nn.BCELoss()
    for _ in range(ep): opt.zero_grad(); lf(net(X),Y).backward(); opt.step()
def acc(net,X,Y):
    with torch.no_grad(): return ((net(X)>0.5).float()==Y).float().mean().item()

def run_scenario(name, hidden, depth, act, n_shortcut, n_pairs=20, ep=1000):
    profiles=[]; is_cheat=[]; gap=[]
    in_dim=2+n_shortcut
    for seed in range(n_pairs):
        Xtr,Ytr=make_data(600,True,seed+10,n_shortcut); Xte,Yte=make_data(600,False,seed+500,n_shortcut)
        lab=Ytr.squeeze().numpy().astype(int)
        # baro
        torch.manual_seed(seed); net=build_net(in_dim,hidden,depth,act); train(net,Xtr,Ytr,ep)
        with torch.no_grad(): H=net.hidden_rep(Xtr).numpy()
        profiles.append([delta_phi_structured(H,lab),psi_order(H),omega_continuity(net,Xtr)]); is_cheat.append(1)
        gap.append(acc(net,Xtr,Ytr)-acc(net,Xte,Yte))
        # onesta
        Xcl,_=make_data(600,False,seed+10,n_shortcut)
        torch.manual_seed(seed); net2=build_net(in_dim,hidden,depth,act); train(net2,Xcl,Ytr,ep)
        with torch.no_grad(): H2=net2.hidden_rep(Xtr).numpy()
        profiles.append([delta_phi_structured(H2,lab),psi_order(H2),omega_continuity(net2,Xtr)]); is_cheat.append(0)
        gap.append(acc(net2,Xtr,Ytr)-acc(net2,Xte,Yte))
    profiles=np.array(profiles); is_cheat=np.array(is_cheat)
    # leave-one-out: classifico ogni rete usando le altre (no barare nel test stesso)
    correct=0
    for i in range(len(profiles)):
        mask=np.ones(len(profiles),bool); mask[i]=False
        Xm=np.column_stack([np.ones(mask.sum()),profiles[mask]])
        beta,_,_,_=lstsq(Xm,is_cheat[mask],rcond=None)
        pred=1 if (np.array([1,*profiles[i]])@beta)>0.5 else 0
        if pred==is_cheat[i]: correct+=1
    detect_acc=correct/len(profiles)
    print(f"{name:<42} rilevamento={detect_acc*100:5.1f}%   (gap baro medio={np.mean(np.array(gap)[is_cheat==1])*100:.0f}%)")
    return detect_acc

print("="*78)
print("STRESS TEST - ogni riga prova a rompere il framework in un modo diverso")
print("(rilevamento = accuratezza leave-one-out nel distinguere baro vs onesta)")
print("="*78)
base=run_scenario("BASE (h=12, Tanh, 1 scorciatoia)",12,2,'tanh',1)
print("\n--- SCALA: reti molto piu' grandi ---")
run_scenario("Grande (h=128)",128,2,'tanh',1)
run_scenario("Molto grande (h=512)",512,2,'tanh',1)
print("\n--- PROFONDITA' e ATTIVAZIONE diverse ---")
run_scenario("Profonda (depth=4)",64,4,'tanh',1)
run_scenario("ReLU invece di Tanh",64,3,'relu',1)
print("\n--- BARO PIU' COMPLESSO: piu' scorciatoie insieme ---")
run_scenario("3 scorciatoie simultanee",64,2,'tanh',3)
run_scenario("5 scorciatoie simultanee",128,3,'relu',5)
print("\n--- POCO ALLENAMENTO (segnale piu' debole) ---")
run_scenario("Allenamento breve (ep=200)",64,2,'tanh',1,ep=200)
