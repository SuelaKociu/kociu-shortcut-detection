"""
FRAMEWORK KOCIU - Delta-phi STRUTTURATO DAL COMPITO
Domanda nuova: non 'quanta' interdipendenza, ma se i neuroni organizzano
l'informazione in modo COERENTE col problema.
Misura: separabilita' delle classi nello spazio nascosto + coerenza interna.
Test di sanita' PRIMA. Controllo casuale SEMPRE. Nessun target predefinito.
"""
import torch, torch.nn as nn, numpy as np
torch.manual_seed(0); np.random.seed(0)

def make_spirals(n=400):
    n_per=n//2; theta=np.sqrt(np.random.rand(n_per))*3*np.pi
    xa=np.stack([theta*np.cos(theta),theta*np.sin(theta)],1)+np.random.randn(n_per,2)*0.3
    xb=np.stack([-theta*np.cos(theta),-theta*np.sin(theta)],1)+np.random.randn(n_per,2)*0.3
    X=np.concatenate([xa,xb]).astype(np.float32); X=(X-X.mean(0))/X.std(0)
    Y=np.concatenate([np.zeros(n_per),np.ones(n_per)]).astype(np.float32)
    return torch.tensor(X), torch.tensor(Y).unsqueeze(1)
X,Y=make_spirals(400); labels=Y.squeeze().numpy().astype(int)

class Net(nn.Module):
    def __init__(s,h=12):
        super().__init__(); s.fc1=nn.Linear(2,h); s.fc2=nn.Linear(h,h); s.fc3=nn.Linear(h,1); s.act=nn.Tanh()
    def hidden_rep(s,x): return s.act(s.fc2(s.act(s.fc1(x))))
    def forward(s,x): return torch.sigmoid(s.fc3(s.hidden_rep(x)))

# ---- METRICA: integrazione strutturata dal compito ----
# Idea: nello spazio dei neuroni nascosti, calcolo
#   - coesione: quanto i punti della STESSA classe sono vicini
#   - separazione: quanto le DUE classi sono lontane tra loro
# Integrazione strutturata = separazione / (coesione + separazione)  -> [0,1]
# Alta = i neuroni hanno organizzato l'informazione coerentemente col compito.
def delta_phi_structured(net, X, labels):
    net.eval()
    with torch.no_grad():
        H = net.hidden_rep(X).numpy()
    # normalizzo lo spazio nascosto
    H = (H - H.mean(0)) / (H.std(0) + 1e-9)
    c0 = H[labels==0]; c1 = H[labels==1]
    m0, m1 = c0.mean(0), c1.mean(0)
    # coesione intra-classe (distanza media dei punti dal proprio centro)
    within = (np.mean(np.linalg.norm(c0-m0,axis=1)) + np.mean(np.linalg.norm(c1-m1,axis=1)))/2
    # separazione inter-classe (distanza tra i centri)
    between = np.linalg.norm(m0-m1)
    if (within+between) < 1e-9: return 0.0
    return float(between / (within + between))

# ---- TEST DI SANITA' ----
print("=== TEST DI SANITA' ===")
# A. dati gia' separati perfettamente nello spazio nascosto -> alto
H_perfect = np.zeros((400,12)); H_perfect[labels==0,0]=-10; H_perfect[labels==1,0]=10
class FakeNet:
    def __init__(s,H): s.H=H
    def eval(s): pass
    def hidden_rep(s,x): return torch.tensor(s.H,dtype=torch.float32)
print(f"A. Classi perfettamente separate:  delta_phi = {delta_phi_structured(FakeNet(H_perfect),X,labels):.4f}  (atteso ALTO)")
# B. classi completamente mescolate (rumore) -> basso
H_mixed = np.random.randn(400,12)
print(f"B. Classi mescolate (rumore puro): delta_phi = {delta_phi_structured(FakeNet(H_mixed),X,labels):.4f}  (atteso BASSO ~0)")
print()

# ---- ESPERIMENTO ----
print("=== ESPERIMENTO: 4 condizioni ===\n")
results=[]
# casuale
net_r=Net()
with torch.no_grad(): acc_r=(((net_r(X)>0.5).float())==Y).float().mean().item()
results.append(("1. CASUALE (non allenata)",acc_r,delta_phi_structured(net_r,X,labels)))
# parzialmente allenata (poche epoche)
net_p=Net(); opt=torch.optim.Adam(net_p.parameters(),lr=0.01); lf=nn.BCELoss()
for _ in range(150): opt.zero_grad(); lf(net_p(X),Y).backward(); opt.step()
with torch.no_grad(): acc_p=(((net_p(X)>0.5).float())==Y).float().mean().item()
results.append(("2. PARZIALE (150 epoche)",acc_p,delta_phi_structured(net_p,X,labels)))
# ben allenata
net=Net(); opt=torch.optim.Adam(net.parameters(),lr=0.01)
for _ in range(3000): opt.zero_grad(); lf(net(X),Y).backward(); opt.step()
with torch.no_grad(): acc_t=(((net(X)>0.5).float())==Y).float().mean().item()
results.append(("3. ALLENATA (3000 epoche)",acc_t,delta_phi_structured(net,X,labels)))
# degradata
net_d=Net(); net_d.load_state_dict(net.state_dict())
with torch.no_grad():
    for p in net_d.parameters(): p+=torch.randn_like(p)*0.8
    acc_d=(((net_d(X)>0.5).float())==Y).float().mean().item()
results.append(("4. DEGRADATA (allenata+rumore)",acc_d,delta_phi_structured(net_d,X,labels)))

print(f"{'Condizione':<34}{'Accuratezza':<14}{'Delta-phi strutturato'}")
print("-"*66)
for name,acc,dphi in results:
    print(f"{name:<34}{acc*100:>5.1f}%{'':<8}{dphi:.4f}")
print()
# verifica correlazione tra accuratezza e metrica
accs=np.array([r[1] for r in results]); dphis=np.array([r[2] for r in results])
r_corr=np.corrcoef(accs,dphis)[0,1]
print(f"Correlazione tra ACCURATEZZA e DELTA-PHI strutturato: r = {r_corr:.3f}")
print(">> Se r alto e positivo: la metrica TRACCIA davvero l'apprendimento.")
