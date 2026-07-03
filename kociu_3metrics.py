"""
FRAMEWORK KOCIU - LE ALTRE 3 METRICHE, COSTRUITE RIGOROSAMENTE
Psi (ordine emergente), Theta (connettivita' topologica), Omega (continuita' temporale)
Ogni metrica: test di sanita' PRIMA, poi validazione contro apprendimento reale.
Nessun target. Controllo casuale sempre.
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

class Net(nn.Module):
    def __init__(s,h=12):
        super().__init__(); s.fc1=nn.Linear(2,h); s.fc2=nn.Linear(h,h); s.fc3=nn.Linear(h,1); s.act=nn.Tanh()
    def hidden_rep(s,x): return s.act(s.fc2(s.act(s.fc1(x))))
    def forward(s,x): return torch.sigmoid(s.fc3(s.hidden_rep(x)))

# ============ PSI: ordine emergente (vs rumore) ============
# Quanto la matrice di correlazione dei neuroni e' piu' strutturata del caso.
# Usiamo l'entropia degli autovalori: bassa entropia spettrale = pochi modi dominanti
# = ordine. Confrontata con la stessa quantita' attesa per rumore puro.
def psi_order(net, X):
    net.eval()
    with torch.no_grad(): H=net.hidden_rep(X).numpy()
    H=(H-H.mean(0))/(H.std(0)+1e-9)
    C=np.corrcoef(H.T); C=np.nan_to_num(C)
    eig=np.abs(np.linalg.eigvalsh(C)); eig=eig/eig.sum()
    spectral_entropy=-np.sum(eig*np.log(eig+1e-12))/np.log(len(eig))
    return float(1.0-spectral_entropy)  # 1 = molto ordinato, 0 = come rumore

# ============ THETA: robustezza distribuita ============
# Quanto l'informazione e' distribuita: se tolgo un neurone a caso, quanto regge?
# Misura la NON-fragilita': sistemi integrati degradano dolcemente, non di colpo.
def theta_robust(net, X, Y):
    net.eval()
    with torch.no_grad():
        base=((net(X)>0.5).float()==Y).float().mean().item()
        H=net.hidden_rep(X); H_full=H.clone()
        drops=[]
        for i in range(H.shape[1]):
            Hd=H_full.clone(); Hd[:,i]=0
            out=torch.sigmoid(net.fc3(Hd))
            acc=((out>0.5).float()==Y).float().mean().item()
            drops.append(base-acc)  # quanto perde togliendo neurone i
        drops=np.array(drops)
    # robustezza = 1 - concentrazione del danno (se un solo neurone e' critico -> fragile)
    if drops.sum()<=1e-9: return 1.0  # togliere nulla cambia: max ridondanza
    p=np.abs(drops)/np.abs(drops).sum()
    conc=-np.sum(p*np.log(p+1e-12))/np.log(len(p))  # entropia: alta=distribuito
    return float(conc)

# ============ OMEGA: continuita' temporale ============
# Quanto le rappresentazioni restano coerenti sotto piccole perturbazioni dell'input
# (analogo di stabilita'/memoria: input simili -> stati interni simili in modo regolare)
def omega_continuity(net, X):
    net.eval()
    with torch.no_grad():
        H=net.hidden_rep(X)
        # perturbo gli input di poco e guardo se gli stati interni cambiano in modo PROPORZIONALE
        Xp=X+torch.randn_like(X)*0.05
        Hp=net.hidden_rep(Xp)
        dX=torch.norm(Xp-X,dim=1).numpy()+1e-9
        dH=torch.norm(Hp-H,dim=1).numpy()
        # continuita' = quanto dH e' proporzionale a dX (correlazione): regolarita' della mappa
        if np.std(dH)<1e-9: return 0.0
        r=np.corrcoef(dX,dH)[0,1]
    return float(np.nan_to_num(r))

# ---------------- TEST DI SANITA' ----------------
print("="*60); print("TEST DI SANITA' DELLE 3 METRICHE"); print("="*60)
X,Y=make_spirals(seed=0); labels=Y.squeeze().numpy().astype(int)

# rete ben allenata e rete casuale per confronto base
torch.manual_seed(0); net_t=Net(); opt=torch.optim.Adam(net_t.parameters(),lr=0.01); lf=nn.BCELoss()
for _ in range(3000): opt.zero_grad(); lf(net_t(X),Y).backward(); opt.step()
torch.manual_seed(0); net_r=Net()

print("\nPSI (ordine emergente):")
print(f"  allenata={psi_order(net_t,X):.3f}  casuale={psi_order(net_r,X):.3f}")
print("\nTHETA (robustezza distribuita):")
print(f"  allenata={theta_robust(net_t,X,Y):.3f}  casuale={theta_robust(net_r,X,Y):.3f}")
print("\nOMEGA (continuita' temporale):")
print(f"  allenata={omega_continuity(net_t,X):.3f}  casuale={omega_continuity(net_r,X):.3f}")

# ---------------- VALIDAZIONE: ogni metrica vs apprendimento ----------------
print("\n"+"="*60); print("VALIDAZIONE: ogni metrica vs ACCURATEZZA (30 reti)"); print("="*60)
accs=[]; PSI=[]; THE=[]; OME=[]
for seed in range(30):
    torch.manual_seed(seed)
    Xs,Ys=make_spirals(seed=seed)
    net=Net(); opt=torch.optim.Adam(net.parameters(),lr=0.01); lf=nn.BCELoss()
    n_ep=int(np.random.RandomState(seed).choice([0,20,50,100,200,500,1500,3000]))
    for _ in range(n_ep): opt.zero_grad(); lf(net(Xs),Ys).backward(); opt.step()
    with torch.no_grad(): acc=((net(Xs)>0.5).float()==Ys).float().mean().item()
    accs.append(acc); PSI.append(psi_order(net,Xs)); THE.append(theta_robust(net,Xs,Ys)); OME.append(omega_continuity(net,Xs))
accs=np.array(accs)
for name,vals in [("PSI (ordine)",PSI),("THETA (robustezza)",THE),("OMEGA (continuita')",OME)]:
    r,p=stats.pearsonr(accs,np.array(vals))
    print(f"  {name:<22} r vs accuratezza = {r:+.3f}  (p={p:.2e})")
