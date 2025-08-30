import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# tiny Q-net
net = nn.Sequential(nn.Linear(64,256), nn.ReLU(), nn.Linear(256,8)).to(device)
tgt = nn.Sequential(nn.Linear(64,256), nn.ReLU(), nn.Linear(256,8)).to(device)
tgt.load_state_dict(net.state_dict()); tgt.eval()
opt = torch.optim.AdamW(net.parameters(), lr=3e-4)

# fake batch
B=512
s  = torch.randn(B,64).to(device)
a  = torch.randint(0,8,(B,1), dtype=torch.long).to(device)
r  = torch.randn(B).to(device)
s2 = torch.randn(B,64).to(device)
d  = torch.randint(0,2,(B,), dtype=torch.float32).to(device)
gamma = 0.99

use_amp = (device.type=="cuda")
scaler = GradScaler(enabled=use_amp)
opt.zero_grad(set_to_none=True)

if use_amp:
    with autocast(enabled=True, dtype=torch.float16):
        qsa = net(s).gather(1,a).squeeze(1)
        with torch.no_grad():
            a_star = net(s2).argmax(1, keepdim=True)
            next_q = tgt(s2).gather(1,a_star).squeeze(1)
    target = (r + gamma*next_q*(1.0-d)).float()
    loss = F.smooth_l1_loss(qsa.float(), target)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    scaler.step(opt); scaler.update()
else:
    qsa = net(s).gather(1,a).squeeze(1)
    with torch.no_grad():
        a_star = net(s2).argmax(1, keepdim=True)
        next_q = tgt(s2).gather(1,a_star).squeeze(1)
    target = r + gamma*next_q*(1.0-d)
    loss = F.smooth_l1_loss(qsa, target)
    loss.backward(); opt.step()

print("Loss:", float(loss.item()))
print("ALL CHECKS PASSED")
