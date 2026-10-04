"""GFAP Spatial Moment Reconstruction -- from-scratch CPU solution.

    python3 solution.py <public_dir> <submission_out.csv>

Approach
--------
Every target value is, by construction, a sum of per-cell contributions over
the cells whose centres fall inside a territory rectangle.  Each of the 64
numbers is therefore an *integral of a per-cell burden field over a rectangle*.
So the network never needs to see the territory program: a convolutional
encoder-decoder, trained from random initialisation, maps one microscopy field
to four non-negative burden-density maps, and the published rectangle geometry
is applied afterwards as an exact separable integration.  One field is encoded
once and reused by all four of its spatial programs, and because integration is
additive the depth-4 hierarchy reconstructs consistently: parent = left + right
holds by construction at every node.

Training signal
---------------
Sixteen coarse sums per program is a weak signal for a dense map.  But a
field's four programs partition it four different ways, and overlaying them
gives a common refinement about forty times finer than any single territory.
Iterative proportional fitting on that refinement recovers the maximum-entropy
burden field consistent with all four programs simultaneously, which supervises
the density maps directly during the early epochs.  That scaffold is annealed
away; the exact rectangle-integral objective, which mirrors the published
scoring formula, then drives the rest of training.

The learned density field is the sole predictor.  Everything is trained from
random initialisation on the released training data only.
"""
import sys, os, json, time, collections
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

H, W, S = 288, 384, 4
GH, GW = H // S, W // S
CH = ["count", "footprint", "elongation", "contrast"]
MU = [1.0, 1.2334, 1.1447, 0.7303]      # mean per-cell burden per unit count
DEPTH_W = [0.20] + [0.35] * 2 + [0.50] * 4 + [0.75] * 8 + [1.00] * 16

BUDGET_S = float(os.environ.get("GFAP_BUDGET_S", 9000))   # wall-clock ceiling
EP_MAX = int(os.environ.get("GFAP_EPOCHS", 70))
EP_MIN = 16
MAX_MODELS = int(os.environ.get("GFAP_MODELS", 3))
ROLL = int(os.environ.get("GFAP_ROLL", 12))   # translation-aug amplitude in pixels
THREADS = int(os.environ.get("GFAP_THREADS", 0)) or max(1, os.cpu_count() or 4)
T0 = time.time()


def log(*a):
    print("[%6.0fs]" % (time.time() - T0), *a, flush=True)


# ============================ network ============================
def cbr(i, o, k=3, s=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, bias=False), nn.BatchNorm2d(o), nn.SiLU(inplace=True))


class Res(nn.Module):
    def __init__(s, c):
        super().__init__()
        s.a = cbr(c, c)
        s.b = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c))
        s.act = nn.SiLU(inplace=True)
    def forward(s, x): return s.act(x + s.b(s.a(x)))


class DensityNet(nn.Module):
    """Field image -> four non-negative burden-density maps at stride 4.

    Channel 0 is cell-count density; the other three are that density times a
    locally predicted per-cell ratio, since footprint, elongation and contrast
    burden are all accumulated over the same cells.
    """
    def __init__(s, cin=3, c4=24, c8=56, c16=96, c32=128, nb=(1, 2, 3, 2), dec=(48, 28), stem=16):
        super().__init__()
        # "How many" is predicted globally, "where" as a normalised spatial
        # distribution.  The per-territory relative-error objective divides by
        # (true + floor), which pulls an integrated density systematically low
        # and corrupts the field total; separating the two removes that pull.
        s.gpool = nn.Sequential(nn.Linear(c32 + c16, 96), nn.SiLU(inplace=True), nn.Linear(96, 4))
        s.gpool[-1].bias.data = torch.tensor([13.4, 16.5, 15.3, 9.6])
        nn.init.normal_(s.gpool[-1].weight, 0, 0.01)
        s.stem = nn.Sequential(cbr(cin, stem, 5, 2), cbr(stem, c4, 3, 2))
        s.e1 = nn.Sequential(*[Res(c4) for _ in range(nb[0])])
        s.e2 = nn.Sequential(cbr(c4, c8, 3, 2), *[Res(c8) for _ in range(nb[1])])
        s.e3 = nn.Sequential(cbr(c8, c16, 3, 2), *[Res(c16) for _ in range(nb[2])])
        s.e4 = nn.Sequential(cbr(c16, c32, 3, 2), *[Res(c32) for _ in range(nb[3])])
        d8, d4 = dec
        s.p4 = nn.Conv2d(c32, c16, 1); s.p3 = nn.Conv2d(c16, c16, 1); s.u3 = cbr(c16, d8)
        s.p2 = nn.Conv2d(c8, d8, 1); s.u2 = cbr(d8, d4)
        s.p1 = nn.Conv2d(c4, d4, 1); s.u1 = cbr(d4, d4)
        s.head = nn.Conv2d(d4, 4, 1)
        nn.init.constant_(s.head.bias, -4.0)
        nn.init.normal_(s.head.weight, 0, 0.01)
        s.register_buffer("mu", torch.tensor(MU))

    def forward(s, x):
        x1 = s.e1(s.stem(x)); x2 = s.e2(x1); x3 = s.e3(x2); x4 = s.e4(x3)
        u = up2(s.p4(x4)) + s.p3(x3)
        u = s.u3(u)
        u = up2(u) + s.p2(x2)
        u = s.u2(u)
        u = up2(u) + s.p1(x1)
        h = s.head(s.u1(u))
        cnt = F.softplus(h[:, :1])
        rat = s.mu[1:].view(1, 3, 1, 1) * torch.exp(0.6 * torch.tanh(h[:, 1:]))
        d = torch.cat([cnt, cnt * rat], 1)
        g = torch.cat([x4.mean((-2, -1)), x3.mean((-2, -1))], 1)
        tot = F.softplus(s.gpool(g)).unsqueeze(-1).unsqueeze(-1)
        return d * tot / d.sum((-2, -1), keepdim=True).clamp_min(1e-6)


def up2(x):
    """Exact 2x spatial upsample (every decoder stage doubles exactly)."""
    return x.repeat_interleave(2, -2).repeat_interleave(2, -1)


def integrate(dens, wy, wx):
    """Exact separable rectangle integration -> (B,P,16,C)."""
    B, C, gh, gw = dens.shape
    P, K = wy.shape[1], wy.shape[2]
    t = torch.bmm(wy.reshape(B, P * K, gh), dens.permute(0, 2, 1, 3).reshape(B, gh, C * gw))
    return (t.view(B, P * K, C, gw) * wx.reshape(B, P * K, 1, gw)).sum(-1).view(B, P, K, C)


def build_tree_t(leaf):
    lv = [leaf]; cur = leaf
    for _ in range(4):
        cur = cur[..., 0::2, :] + cur[..., 1::2, :]
        lv.append(cur)
    return torch.cat(lv[::-1], dim=-2)


def metric_loss(pl, tl, dw, eps=1e-6):
    """Mirrors the published score: root term, depth-weighted tree term,
    terminal-distribution term, at the published weights."""
    P, T = build_tree_t(pl), build_tree_t(tl)
    rt = T[:, 0, :]
    root_err = (P[:, 0, :] - rt).abs() / (rt + 1.0)
    flo = 0.35 * rt / 16.0 + 0.25
    tree_err = (((P - T).abs() / (T + flo.unsqueeze(1))) * dw).sum(1) / dw.sum()
    p = pl / (pl.sum(1, keepdim=True) + eps)
    q = tl / (tl.sum(1, keepdim=True) + eps)
    tv = 0.5 * (p - q).abs().sum(1)
    return 0.20 * root_err.mean() + 0.50 * tree_err.mean() + 0.30 * tv.mean()


# ================= fusing the four programs into dense targets =================
def overlay_grid(progs):
    xs, ys = set(), set()
    for p in progs:
        for n in p:
            xs.add(n["x_min"]); xs.add(n["x_max"]); ys.add(n["y_min"]); ys.add(n["y_max"])
    xs = np.array(sorted(xs)); ys = np.array(sorted(ys))
    GX, GY = np.meshgrid(0.5 * (xs[:-1] + xs[1:]), 0.5 * (ys[:-1] + ys[1:]))
    idx = np.zeros((len(progs), GY.shape[0], GX.shape[1]), np.int64)
    for k, p in enumerate(progs):
        for li, n in enumerate(p[15:31]):
            m = (GX >= n["x_min"]) & (GX < n["x_max"]) & (GY >= n["y_min"]) & (GY < n["y_max"])
            idx[k][m] = li
    return xs, ys, idx


def ipf(idx, targets, iters=200, eps=1e-12):
    """Max-entropy field matching all four programs' terminal sums at once."""
    K, ny, nx = idx.shape
    C = targets.shape[-1]
    x = np.ones((ny, nx, C))
    flat = idx.reshape(K, -1)
    for _ in range(iters):
        for k in range(K):
            cur = np.zeros((16, C))
            np.add.at(cur, flat[k], x.reshape(-1, C))
            sc = np.where(cur > eps, targets[k] / np.maximum(cur, eps), 0.0)
            x = (x.reshape(-1, C) * sc[flat[k]]).reshape(ny, nx, C)
    return x


def _amat(edges, n):
    e = np.arange(n + 1) * S
    return np.clip(np.minimum(edges[1:, None], e[None, 1:]) - np.maximum(edges[:-1, None], e[None, :-1]), 0, None)


def make_targets(fields, byf):
    out = np.zeros((len(fields), 4, GH, GW), np.float32)
    for i, f in enumerate(fields):
        rs = byf[f]
        T = np.array([r["target_leaf_moments"] for r in rs], np.float64)
        xs, ys, idx = overlay_grid([r["program_nodes"] for r in rs])
        x = ipf(idx, T)
        Ay = _amat(ys.astype(float), GH); Ax = _amat(xs.astype(float), GW)
        fy = Ay / np.maximum(Ay.sum(1, keepdims=True), 1e-12)
        fx = Ax / np.maximum(Ax.sum(1, keepdims=True), 1e-12)
        out[i] = np.einsum("yxc,yg,xh->cgh", x, fy, fx)
    return out


# ============================ data ============================
def overlap_1d(lo, hi, n):
    e = np.arange(n + 1) * S
    return np.clip(np.minimum(hi, e[1:]) - np.maximum(lo, e[:-1]), 0, None) / S


def load_split(pub, split, need_y):
    rows = [json.loads(l) for l in open(os.path.join(pub, split + ".jsonl"), encoding="utf-8")]
    byf = collections.OrderedDict()
    for r in rows:
        byf.setdefault(r["field_id"], []).append(r)
    fields = list(byf)
    fmap = {f: i for i, f in enumerate(fields)}
    imgs = np.zeros((len(fields), H, W), np.uint8)
    mfield = np.zeros(len(fields), np.float32)
    for f in fields:
        a = np.array(Image.open(os.path.join(pub, byf[f][0]["image_file"])).convert("L"))
        if a.shape != (H, W):
            a = np.array(Image.fromarray(a).resize((W, H), Image.BILINEAR))
        imgs[fmap[f]] = a
        nz = a[a > 0]
        mfield[fmap[f]] = np.median(nz) if nz.size else 0.0
    N = len(rows)
    wy = np.zeros((N, 16, GH), np.float32); wx = np.zeros((N, 16, GW), np.float32)
    Y = np.zeros((N, 16, 4), np.float32) if need_y else None
    fid = np.zeros(N, np.int64); sids = []
    for i, r in enumerate(rows):
        for l, n in enumerate(r["program_nodes"][15:31]):
            wy[i, l] = overlap_1d(n["y_min"], n["y_max"], GH)
            wx[i, l] = overlap_1d(n["x_min"], n["x_max"], GW)
        if need_y:
            Y[i] = np.array(r["target_leaf_moments"], np.float32)
        fid[i] = fmap[r["field_id"]]; sids.append(r["sample_id"])
    rowsof = np.zeros((len(fields), 4), np.int64); c = np.zeros(len(fields), np.int64)
    for i, f in enumerate(fid):
        rowsof[f, c[f]] = i; c[f] += 1
    return dict(byf=byf, fields=fields, imgs=imgs, mfield=mfield, wy=wy, wx=wx,
                Y=Y, fid=fid, sids=sids, rowsof=rowsof)


def make_input(imgs, mfield):
    """Field-relative intensity (the unit the contrast burden is defined in),
    absolute level, and the valid-content mask left by letterboxing."""
    x = imgs.float(); m = (x > 0).float()
    return torch.stack([(x - mfield.view(-1, 1, 1)) / 96.0 * m, (x / 255.0 - 0.5) * 2 * m, m], 1)


# ============================ train / predict ============================
def train_one(d, dense, epochs, seed, lr=3e-3, bs=8, wdense=1.0, dense_ep=18):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = DensityNet()
    imgs = torch.from_numpy(d["imgs"]); mf = torch.from_numpy(d["mfield"])
    wy = torch.from_numpy(d["wy"]); wx = torch.from_numpy(d["wx"])
    Y = torch.from_numpy(d["Y"]); de = torch.from_numpy(dense)
    ro = d["rowsof"]; nF = len(d["fields"])
    dw = torch.tensor(DEPTH_W).view(1, 31, 1)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, lr, total_steps=epochs * int(np.ceil(nF / bs)), pct_start=0.25)
    for ep in range(epochs):
        net.train(); perm = rng.permutation(nF); t0 = time.time(); tot = 0.0
        wd_ = wdense * max(0.0, 1 - ep / max(1, dense_ep))
        for b in range(0, nF, bs):
            i = perm[b:b + bs]
            im = imgs[i].clone(); r = ro[i]
            byi, bxi, dei = wy[r].clone(), wx[r].clone(), de[i].clone()
            fh = torch.from_numpy(rng.random(len(i)) < 0.5)
            fv = torch.from_numpy(rng.random(len(i)) < 0.5)
            if fh.any():
                im[fh] = torch.flip(im[fh], [-1]); bxi[fh] = torch.flip(bxi[fh], [-1]); dei[fh] = torch.flip(dei[fh], [-1])
            if fv.any():
                im[fv] = torch.flip(im[fv], [-2]); byi[fv] = torch.flip(byi[fv], [-1]); dei[fv] = torch.flip(dei[fv], [-2])
            if ROLL:
                # Circular shift by a whole number of grid cells.  Shifting the
                # image and the territory weights together leaves every leaf sum
                # exactly unchanged, so this augments without perturbing labels.
                k = ROLL // S
                for b_ in range(len(i)):
                    dy, dx = int(rng.integers(-k, k + 1)), int(rng.integers(-k, k + 1))
                    if dy or dx:
                        im[b_] = torch.roll(im[b_], (dy * S, dx * S), (-2, -1))
                        byi[b_] = torch.roll(byi[b_], dy, -1)
                        bxi[b_] = torch.roll(bxi[b_], dx, -1)
                        dei[b_] = torch.roll(dei[b_], (dy, dx), (-2, -1))
            x = make_input(im, mf[i])
            x[:, :2] += torch.from_numpy(rng.normal(0, 0.02, x[:, :2].shape).astype(np.float32)) * x[:, 2:3]
            dens = net(x)
            pl = integrate(dens, byi, bxi).reshape(-1, 16, 4)
            loss = metric_loss(pl, Y[r].reshape(-1, 16, 4), dw)
            if wd_ > 0:
                loss = loss + wd_ * (dens - dei).abs().mean() * 300.0
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step(); sch.step(); tot += float(loss) * len(i)
        if ep % 5 == 0 or ep == epochs - 1:
            log("   seed%d ep%02d/%d loss %.4f (%.0fs/ep)" % (seed, ep + 1, epochs, tot / nF, time.time() - t0))
    return net


@torch.no_grad()
def predict_dens(nets, d, ebs=16):
    imgs = torch.from_numpy(d["imgs"]); mf = torch.from_numpy(d["mfield"])
    out = []
    for b in range(0, len(d["fields"]), ebs):
        sl = slice(b, b + ebs)
        x = make_input(imgs[sl], mf[sl]); acc = 0
        for net in nets:
            net.eval()
            v = net(x)
            v = v + torch.flip(net(torch.flip(x, [-1])), [-1])
            v = v + torch.flip(net(torch.flip(x, [-2])), [-2])
            v = v + torch.flip(net(torch.flip(x, [-1, -2])), [-1, -2])
            acc = acc + v / 4.0
        out.append((acc / len(nets)).numpy())
    return np.concatenate(out)


def dens_to_leaves(dens, d):
    ro = d["rowsof"]
    P = integrate(torch.from_numpy(dens), torch.from_numpy(d["wy"][ro]),
                  torch.from_numpy(d["wx"][ro])).numpy()
    out = np.zeros((len(d["sids"]), 16, 4), np.float32)
    out[ro.ravel()] = P.reshape(-1, 16, 4)
    return out


# ============================ output calibration ============================
# (deadzone, exponent, gain) per channel.  The score divides each error by
# (true + floor), so an over-stated empty territory costs far more than an
# under-stated occupied one; the score-optimal point prediction is therefore
# not the conditional mean.  Fitted out-of-fold against the published formula.
CAL = np.array([[0.194, 0.875, 1.034],
                [0.159, 0.875, 1.014],
                [0.229, 0.875, 1.014],
                [0.124, 0.950, 1.014]], np.float64)


def calibrate(P, cal=CAL):
    Q = np.empty(P.shape, np.float64)
    for c in range(P.shape[2]):
        dz, ex, gn = cal[c]
        p = P[:, :, c].astype(np.float64)
        tot = p.sum(1, keepdims=True)
        q = np.clip(p - dz * tot / 16.0, 0, None) ** ex
        s = q.sum(1, keepdims=True)
        Q[:, :, c] = np.where(s > 0, q * (gn * tot / np.maximum(s, 1e-12)), gn * tot / 16.0)
    return Q


# ============================ main ============================
def plan_epochs(d, dense):
    """Time a few real steps so the run fits the wall-clock budget."""
    net = DensityNet()
    opt = torch.optim.AdamW(net.parameters(), 1e-4)
    imgs = torch.from_numpy(d["imgs"][:8]); mf = torch.from_numpy(d["mfield"][:8])
    x = make_input(imgs, mf)
    r = d["rowsof"][:8]
    wy = torch.from_numpy(d["wy"][r]); wx = torch.from_numpy(d["wx"][r])
    Y = torch.from_numpy(d["Y"][r]).reshape(-1, 16, 4)
    dw = torch.tensor(DEPTH_W).view(1, 31, 1)
    for _ in range(2):
        loss = metric_loss(integrate(net(x), wy, wx).reshape(-1, 16, 4), Y, dw)
        opt.zero_grad(); loss.backward(); opt.step()
    t = time.time(); n = 3
    for _ in range(n):
        loss = metric_loss(integrate(net(x), wy, wx).reshape(-1, 16, 4), Y, dw)
        opt.zero_grad(); loss.backward(); opt.step()
    per_step = (time.time() - t) / n
    ep_time = per_step * np.ceil(len(d["fields"]) / 8) * 1.15 + 5
    avail = BUDGET_S - (time.time() - T0) - 420
    ep = int(min(EP_MAX, max(EP_MIN, avail // max(ep_time, 1e-6))))
    log("timing probe: ~%.0fs/epoch -> %d epochs per model" % (ep_time, ep))
    return ep


def main():
    pub, outp = sys.argv[1], sys.argv[2]
    torch.set_num_threads(THREADS)
    log("threads %d, budget %.0fs" % (THREADS, BUDGET_S))

    tr = load_split(pub, "train", True)
    te = load_split(pub, "test", False)
    log("train: %d fields / %d rows | test: %d fields / %d rows"
        % (len(tr["fields"]), len(tr["sids"]), len(te["fields"]), len(te["sids"])))

    log("fusing each field's four programs into a dense burden target ...")
    dense = make_targets(tr["fields"], tr["byf"])
    log("dense targets ready", dense.shape)

    ep = plan_epochs(tr, dense)
    # Train independently initialised models for as long as the clock allows:
    # averaging their density fields mainly reduces field-total variance, which
    # is where most of the remaining error sits.
    nets = []
    while len(nets) < MAX_MODELS:
        t_start = time.time()
        nets.append(train_one(tr, dense, ep, seed=100 + len(nets), dense_ep=min(18, max(4, ep // 3))))
        took = time.time() - t_start
        left = BUDGET_S - (time.time() - T0)
        log("model %d trained in %.0fs; %.0fs of budget left" % (len(nets), took, left))
        if left < took * 1.05 + 360:
            break

    log("predicting test fields ...")
    raw = dens_to_leaves(predict_dens(nets, te), te)
    try:
        np.save(os.path.splitext(outp)[0] + "_raw.npy", raw)
    except Exception:
        pass
    P = calibrate(raw)
    P = np.clip(np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)

    cols = ["sample_id"] + ["leaf_%02d_%s" % (i + 1, c) for i in range(16) for c in CH]
    ss = os.path.join(pub, "sample_submission.csv")
    if os.path.exists(ss):
        with open(ss) as f:
            hdr = f.readline().strip().split(",")
        if len(hdr) == 65:
            cols = hdr
    flat = P.reshape(len(P), 64)
    with open(outp, "w", newline="") as f:
        f.write(",".join(cols) + "\n")
        for i, sid in enumerate(te["sids"]):
            f.write(sid + "," + ",".join("%.6f" % v for v in flat[i]) + "\n")
    log("wrote %s  rows=%d cols=%d" % (outp, len(flat), len(cols)))


if __name__ == "__main__":
    main()
