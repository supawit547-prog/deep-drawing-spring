"""
dd_core.py — ส่วนคำนวณหลักของโปรแกรมทำนายค่าสปริงแม่พิมพ์ลากขึ้นรูป
(ฟิสิกส์ / สร้างข้อมูลจำลอง / เทรนโมเดล ML / ออกแบบสปริง)

ใช้ร่วมกับ app.py (Streamlit) หรือ import ไปใช้ในสคริปต์อื่นก็ได้
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# ---------------------------------------------------------------------------
# ข้อมูลวัสดุ (ค่าทั่วไป — ควรแก้ตามใบรับรองวัสดุที่โรงงานใช้จริง)
# c    = ค่าคงที่ในสูตร Siebel (2–3)
# bmax = อัตราส่วนการลากสูงสุดโดยประมาณ (ลากครั้งแรก)
# ---------------------------------------------------------------------------
MATERIALS: dict[str, dict] = {
    "SPCC":   {"name": "SPCC (เหล็กแผ่นรีดเย็น)",   "Rm": 330, "Re": 210, "r": 1.4, "c": 2.5, "bmax": 2.0},
    "SPCE":   {"name": "SPCE / DC04 (เหล็กลากลึก)", "Rm": 300, "Re": 170, "r": 1.8, "c": 2.5, "bmax": 2.1},
    "SPHC":   {"name": "SPHC (เหล็กรีดร้อน)",       "Rm": 340, "Re": 230, "r": 1.0, "c": 2.5, "bmax": 1.9},
    "SUS304": {"name": "SUS304 (สแตนเลส)",          "Rm": 600, "Re": 260, "r": 1.0, "c": 3.0, "bmax": 2.0},
    "SUS430": {"name": "SUS430 (สแตนเลส)",          "Rm": 480, "Re": 290, "r": 1.2, "c": 2.8, "bmax": 1.9},
    "A1100":  {"name": "อะลูมิเนียม A1100-O",        "Rm": 95,  "Re": 35,  "r": 0.8, "c": 2.0, "bmax": 2.0},
    "A5052":  {"name": "อะลูมิเนียม A5052-H32",      "Rm": 230, "Re": 170, "r": 0.7, "c": 2.0, "bmax": 1.8},
    "C2680":  {"name": "ทองเหลือง C2680",            "Rm": 350, "Re": 150, "r": 0.9, "c": 2.2, "bmax": 2.1},
    "C1100":  {"name": "ทองแดง C1100",               "Rm": 230, "Re": 80,  "r": 0.9, "c": 2.2, "bmax": 2.1},
    "custom": {"name": "กำหนดเอง",                   "Rm": 350, "Re": 220, "r": 1.2, "c": 2.5, "bmax": 2.0},
}

# คลาสสปริงแม่พิมพ์: ขีดจำกัดระยะยุบ (% ของความยาวอิสระ) เป็นค่าประมาณทั่วไป
CLASSES = [
    {"id": "L", "name": "งานเบา (Light)",        "life": 30, "max": 50, "iso": "green",  "jis": "yellow"},
    {"id": "M", "name": "งานกลาง (Medium)",      "life": 25, "max": 40, "iso": "blue",   "jis": "blue"},
    {"id": "H", "name": "งานหนัก (Heavy)",       "life": 20, "max": 30, "iso": "red",    "jis": "red"},
    {"id": "X", "name": "งานหนักพิเศษ (Extra)",  "life": 17, "max": 25, "iso": "yellow", "jis": "green"},
]
COLOR_TH = {"green": "เขียว", "blue": "น้ำเงิน", "red": "แดง", "yellow": "เหลือง"}
COLOR_HEX = {"green": "#2E8B57", "blue": "#2F64B5", "red": "#C23B34", "yellow": "#D9A914"}

FEATURES = ["t", "D0", "dp", "rd", "h", "Rm", "Re", "r", "mu"]
TARGETS = ["F_BH_kN", "F_draw_kN"]
N_LOG_COLS = 7  # 7 คอลัมน์แรกแปลงเป็น log (ค่าทางเรขาคณิต/ความแข็งแรง)

FEATURE_TH = {
    "t": "ความหนา t", "D0": "แผ่นเปล่า D0", "dp": "พันช์ dp", "rd": "รัศมีดาย rd",
    "h": "ความลึก h", "Rm": "Rm", "Re": "Re", "r": "ค่า r", "mu": "แรงเสียดทาน μ",
}


# ---------------------------------------------------------------------------
# ฟิสิกส์ (สูตรเชิงประสบการณ์)
# ---------------------------------------------------------------------------
def physics(x, c=2.5, bmax=2.0, extras: bool = False) -> dict:
    """คำนวณแรงกดยึด (Siebel) และแรงลาก — รับ dict หรือ DataFrame (vectorized)

    p [MPa] = c·10⁻³·[(β−1)² + D0/(200·t)]·Rm
    F_BH    = p · A_flange
    F_draw  = π·(dp+t)·t·Rm·n + 2·μ·F_BH
    """
    g = {k: np.asarray(x[k], dtype=float) for k in FEATURES}
    t, D0, dp, rd, Rm, Re, r, mu = g["t"], g["D0"], g["dp"], g["rd"], g["Rm"], g["Re"], g["r"], g["mu"]
    c = np.asarray(c, dtype=float)
    bmax = np.asarray(bmax, dtype=float)

    beta = D0 / dp
    p = c * 1e-3 * ((beta - 1) ** 2 + D0 / (200 * t)) * Rm
    if extras:  # ผลกระทบเพิ่มเติมที่ใช้ตอนสร้างข้อมูลจำลอง
        p = p * (2.6 / (1 + r)) ** 0.25 * ((Re / Rm) / 0.6) ** 0.3 * (1 + 0.8 * (t / rd - 0.15))
    di = dp + 2 * rd + 2 * t
    area = np.pi / 4 * (D0 ** 2 - di ** 2)
    fbh = np.maximum(p * area, 0) / 1000.0  # kN
    n = np.clip(1.2 * (beta - 1) / (bmax - 1), 0.3, 1.25)
    fd = np.pi * (dp + t) * t * Rm * n / 1000.0 + 2 * mu * fbh * (1.1 if extras else 1.0)
    return {"F_BH_kN": fbh, "F_draw_kN": fd, "beta": beta, "area": area}


def blank_diameter(dp: float, h: float, rp: float = 0.0, t: float = 0.0) -> float:
    """เส้นผ่านศูนย์กลางแผ่นเปล่าของถ้วยทรงกระบอก (พื้นที่ผิวคงที่)
    D0 = √(d² + 4·d·h − 1.72·d·r − 0.56·r²)   โดย d = dp + t (เส้นผ่านศูนย์กลางกลางเนื้อ), r = รัศมีก้นถ้วย"""
    d = dp + t
    val = d * d + 4 * d * h - 1.72 * d * rp - 0.56 * rp * rp
    return math.sqrt(max(val, d * d))


# ค่าเพิ่มเติมต่อวัสดุ: k ในสูตรระยะช่องว่าง (Oehler) และอัตราส่วนการลากครั้งถัดไป
MAT_EXTRA = {
    "SPCC": (0.07, 1.25), "SPCE": (0.07, 1.28), "SPHC": (0.07, 1.22),
    "SUS304": (0.07, 1.20), "SUS430": (0.07, 1.20),
    "A1100": (0.02, 1.25), "A5052": (0.02, 1.20),
    "C2680": (0.04, 1.30), "C1100": (0.04, 1.30), "custom": (0.07, 1.25),
}


def die_clearance(t: float, mat_key: str) -> float:
    """ระยะช่องว่างพันช์-ดาย ต่อข้าง: c = t + k·√(10·t)"""
    k = MAT_EXTRA.get(mat_key, (0.07, 1.25))[0]
    return t + k * math.sqrt(10 * t)


def die_radius_recommend(D0: float, dp: float, t: float) -> float:
    """รัศมีขอบดายแนะนำ (Kaczmarek): rd = 0.8·√((D0 − dp)·t)"""
    return 0.8 * math.sqrt(max(D0 - dp, 0) * t)


def draw_stages(D0: float, dp: float, mat_key: str) -> list[float]:
    """เส้นผ่านศูนย์กลางแต่ละขั้นการลาก: d1 = D0/βmax, dn = d(n−1)/β_next จนถึง dp"""
    bmax = MATERIALS.get(mat_key, MATERIALS["custom"])["bmax"]
    bnext = MAT_EXTRA.get(mat_key, (0.07, 1.25))[1]
    stages, d = [], D0 / bmax
    while d > dp and len(stages) < 10:
        stages.append(d)
        d = d / bnext
    stages.append(dp)
    return stages


# ---------------------------------------------------------------------------
# ข้อมูลจำลอง
# ---------------------------------------------------------------------------
def generate_synthetic(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keys = [k for k in MATERIALS if k != "custom"]
    m = n * 3
    mk = rng.choice(keys, m)
    base = pd.DataFrame([MATERIALS[k] for k in mk])

    Rm = base["Rm"].to_numpy() * rng.uniform(0.92, 1.08, m)
    Re = np.minimum(base["Re"].to_numpy() * rng.uniform(0.88, 1.12, m), 0.95 * Rm)
    r = base["r"].to_numpy() * rng.uniform(0.85, 1.15, m)
    t = np.exp(rng.uniform(np.log(0.4), np.log(4.0), m))
    dp = np.exp(rng.uniform(np.log(15), np.log(250), m))
    beta = rng.uniform(1.35, base["bmax"].to_numpy())
    D0 = beta * dp
    rd = t * rng.uniform(4, 10, m)
    mu = rng.uniform(0.05, 0.15, m)
    hmax = (D0 ** 2 - dp ** 2) / (4 * dp)
    h = hmax * rng.uniform(0.5, 1.0, m)

    df = pd.DataFrame({"t": t, "D0": D0, "dp": dp, "rd": rd, "h": h,
                       "Rm": Rm, "Re": Re, "r": r, "mu": mu, "material": mk})
    ph = physics(df, base["c"].to_numpy(), base["bmax"].to_numpy(), extras=True)
    df["F_BH_kN"] = ph["F_BH_kN"] * np.exp(rng.normal(0, 0.08, m))
    df["F_draw_kN"] = ph["F_draw_kN"] * np.exp(rng.normal(0, 0.05, m))
    ok = (D0 >= dp + 2 * rd + 2 * t + 4) & (df["F_BH_kN"] > 0.02)
    return df[ok].head(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# ข้อมูลจริงจากโรงงาน
# ---------------------------------------------------------------------------
_ALIAS = {
    "t": "t", "d0": "D0", "dp": "dp", "rd": "rd", "h": "h", "rm": "Rm", "re": "Re", "r": "r", "mu": "mu",
    "f_bh_kn": "F_BH_kN", "fbh": "F_BH_kN", "f_bh": "F_BH_kN",
    "f_draw_kn": "F_draw_kN", "fdraw": "F_draw_kN", "f_draw": "F_draw_kN",
    "material": "material", "note": "note", "date": "date",
}
REAL_COLUMNS = FEATURES + TARGETS + ["material", "note", "date"]


def clean_real(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """เปลี่ยนชื่อคอลัมน์ให้ตรงมาตรฐาน แปลงเป็นตัวเลข ตัดแถวที่ข้อมูลไม่ครบ
    คืนค่า (DataFrame ที่สะอาด, จำนวนแถวที่ถูกตัด)"""
    ren = {}
    for col in df.columns:
        key = "".join(ch for ch in str(col).strip().lower() if ch.isalnum() or ch == "_")
        if key in _ALIAS:
            ren[col] = _ALIAS[key]
    df = df.rename(columns=ren)
    missing = [c for c in FEATURES + ["F_BH_kN"] if c not in df.columns]
    if missing:
        raise ValueError("ไม่พบคอลัมน์: " + ", ".join(missing))
    for c in REAL_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan if c in TARGETS else ""
    for c in FEATURES + TARGETS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    valid = (df[FEATURES + ["F_BH_kN"]] > 0).all(axis=1)
    df.loc[~(df["F_draw_kN"] > 0), "F_draw_kN"] = np.nan
    out = df.loc[valid, REAL_COLUMNS].reset_index(drop=True)
    for c in ["material", "note", "date"]:
        out[c] = out[c].fillna("").astype(str)
    return out, int((~valid).sum())


def load_real(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            df, _ = clean_real(pd.read_csv(path))
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=REAL_COLUMNS)


def save_real(df: pd.DataFrame, path: Path) -> None:
    df[REAL_COLUMNS].to_csv(path, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# โมเดล ML
# ---------------------------------------------------------------------------
MODEL_KINDS = {
    "mlp": "Neural Network (MLP) — แนะนำ",
    "gb": "Gradient Boosting — เหมาะเมื่อข้อมูลจริงมีหลายร้อยแถว",
    "rf": "Random Forest",
    "ridge": "Linear (Ridge) — เหมาะกับข้อมูลจริงน้อย",
}


def log_features(X):
    X = np.asarray(X, dtype=float).copy()
    X[:, :N_LOG_COLS] = np.log(np.clip(X[:, :N_LOG_COLS], 1e-6, None))
    return X


def make_model(kind: str = "gb", seed: int = 0):
    if kind == "rf":
        reg = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    elif kind == "mlp":
        reg = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", learning_rate_init=0.003,
                           max_iter=1500, early_stopping=True, n_iter_no_change=30, random_state=seed)
    elif kind == "ridge":
        reg = RidgeCV(alphas=np.logspace(-3, 3, 13))
    else:
        reg = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06, max_leaf_nodes=31,
                                            l2_regularization=0.1, random_state=seed)
    return make_pipeline(FunctionTransformer(log_features), StandardScaler(), reg)


def baseline(df) -> tuple[np.ndarray, np.ndarray]:
    """ค่าจากสูตร Siebel แบบทั่วไป (c=2.5, βmax=2.0) — โมเดลเรียนรู้ 'ค่าแก้' เทียบกับค่านี้
    (physics-informed residual) ทำให้เทรนด้วยข้อมูลจริงจำนวนน้อยได้ดีขึ้นมาก"""
    ph = physics(df, 2.5, 2.0)
    return np.atleast_1d(ph["F_BH_kN"]), np.atleast_1d(ph["F_draw_kN"])


def build_training_set(synth: pd.DataFrame, real: pd.DataFrame, mode: str) -> pd.DataFrame:
    """mode = 'mix'  : ข้อมูลจำลอง + ข้อมูลจริง (ข้อมูลจริงถูกทำซ้ำเพื่อเพิ่มน้ำหนัก)
       mode = 'real' : ข้อมูลจริงอย่างเดียว"""
    parts = []
    if mode == "mix":
        s = synth[FEATURES + TARGETS].copy()
        s["group"] = [f"S{i}" for i in range(len(s))]
        s["is_real"] = False
        parts.append(s)
    if len(real):
        rr = real[FEATURES + TARGETS].copy().reset_index(drop=True)
        rr["group"] = [f"R{i}" for i in range(len(rr))]
        rr["is_real"] = True
        rep = min(20, max(3, round(400 / len(rr)))) if mode == "mix" else 1
        parts += [rr] * rep
    if not parts:
        raise ValueError("ไม่มีข้อมูลสำหรับเทรน")
    return pd.concat(parts, ignore_index=True)


def train(synth: pd.DataFrame, real: pd.DataFrame, mode: str = "mix", kind: str = "gb", seed: int = 0) -> dict:
    if mode == "real" and len(real) < 15:
        raise ValueError(f"ข้อมูลจริงมี {len(real)} แถว ต้องมีอย่างน้อย 15 แถวสำหรับโหมดข้อมูลจริงอย่างเดียว")
    data = build_training_set(synth, real, mode)
    b_bh, b_fd = baseline(data)
    data = data.assign(base_bh=b_bh, base_fd=b_fd)
    data = data[(data["base_bh"] > 0) & (data["base_fd"] > 0)]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr_idx, te_idx = next(gss.split(data, groups=data["group"]))
    tr, te = data.iloc[tr_idx], data.iloc[te_idx].drop_duplicates(subset="group")

    model_bh = make_model(kind, seed).fit(tr[FEATURES], np.log(tr["F_BH_kN"] / tr["base_bh"]))
    tr_fd = tr.dropna(subset=["F_draw_kN"])
    model_fd = (make_model(kind, seed).fit(tr_fd[FEATURES], np.log(tr_fd["F_draw_kN"] / tr_fd["base_fd"]))
                if len(tr_fd) >= 10 else None)

    te = te.copy()
    te["pred_F_BH_kN"] = te["base_bh"] * np.exp(model_bh.predict(te[FEATURES]))
    metrics = {
        "r2_bh": r2_score(te["F_BH_kN"], te["pred_F_BH_kN"]),
        "mape_bh": mean_absolute_percentage_error(te["F_BH_kN"], te["pred_F_BH_kN"]) * 100,
    }
    if model_fd is not None:
        te_fd = te.dropna(subset=["F_draw_kN"])
        if len(te_fd) >= 3:
            pfd = te_fd["base_fd"] * np.exp(model_fd.predict(te_fd[FEATURES]))
            metrics["r2_fd"] = r2_score(te_fd["F_draw_kN"], pfd)
            metrics["mape_fd"] = mean_absolute_percentage_error(te_fd["F_draw_kN"], pfd) * 100
    te_real = te[te["is_real"]]
    if len(te_real) >= 2:
        metrics["mape_bh_real"] = mean_absolute_percentage_error(te_real["F_BH_kN"], te_real["pred_F_BH_kN"]) * 100
        metrics["n_test_real"] = len(te_real)

    imp_df = te.sample(min(len(te), 400), random_state=seed)
    pi = permutation_importance(model_bh, imp_df[FEATURES], np.log(imp_df["F_BH_kN"] / imp_df["base_bh"]),
                                n_repeats=5, random_state=seed)
    importance = pd.Series(pi.importances_mean, index=FEATURES).sort_values()

    return {
        "model_bh": model_bh, "model_fd": model_fd, "metrics": metrics,
        "test": te[FEATURES + TARGETS + ["pred_F_BH_kN", "is_real"]].reset_index(drop=True),
        "importance": importance,
        "ranges": {c: (float(tr[c].min()), float(tr[c].max())) for c in FEATURES},
        "kind": kind, "mode": mode,
        "n_train": int(tr["group"].nunique()), "n_test": len(te), "n_real": len(real),
    }


def predict(bundle: dict, x: dict) -> tuple[float, float | None]:
    X = pd.DataFrame([{k: x[k] for k in FEATURES}])
    b_bh, b_fd = baseline(X)
    fbh = float(b_bh[0] * np.exp(bundle["model_bh"].predict(X)[0]))
    fd = float(b_fd[0] * np.exp(bundle["model_fd"].predict(X)[0])) if bundle.get("model_fd") is not None else None
    return fbh, fd


# ---------------------------------------------------------------------------
# ออกแบบสปริง
# ---------------------------------------------------------------------------
def spring_design(fbh_kN: float, N: int, s0: float, h: float, margin: float, L0: float, sf: float) -> dict:
    F0_total = fbh_kN * sf * 1000.0          # N ที่ระยะอัดล่วงหน้า
    k = F0_total / N / s0                    # N/mm ต่อตัว
    smax = s0 + h + margin                   # ระยะยุบรวม
    F_end_total = N * k * smax
    defl = smax / L0 * 100
    rows = []
    for c in CLASSES:
        status = "ok" if defl <= c["life"] else ("mid" if defl <= c["max"] else "bad")
        rows.append({**c, "status": status, "min_L0": smax / (c["life"] / 100)})
    best = next((r["id"] for r in reversed(rows) if r["status"] == "ok"), None) \
        or next((r["id"] for r in reversed(rows) if r["status"] == "mid"), None)
    return {
        "k": k, "F0_each": F0_total / N, "F0_total": F0_total, "smax": smax,
        "F_end_each": k * smax, "F_end_total": F_end_total, "defl_pct": defl,
        "ratio": smax / s0, "classes": rows, "best": best,
    }


def check_warnings(x: dict, mat: dict, design: dict, ranges: dict | None) -> list[tuple[str, str]]:
    w: list[tuple[str, str]] = []
    beta = x["D0"] / x["dp"]
    if beta > mat["bmax"]:
        w.append(("error", f"อัตราส่วนการลาก β = {beta:.2f} เกินขีดจำกัดประมาณ {mat['bmax']} ควรแบ่งลากหลายครั้ง"))
    hmax = (x["D0"] ** 2 - x["dp"] ** 2) / (4 * x["dp"])
    if x["h"] > hmax * 1.05:
        w.append(("error", f"ความลึก h เกินความสูงถ้วยที่แผ่นขนาดนี้ให้ได้ (≈ {hmax:.1f} มม.)"))
    if design["ratio"] > 3:
        w.append(("warning", f"แรงกดยึดท้ายจังหวะสูงกว่าตอนเริ่ม {design['ratio']:.1f} เท่า เสี่ยงชิ้นงานขาด — "
                             "เพิ่มระยะอัดล่วงหน้า ใช้สปริงยาวขึ้น หรือใช้ gas spring"))
    if design["best"] is None:
        w.append(("error", f"ระยะยุบ {design['defl_pct']:.0f}% เกินทุกคลาส ต้องใช้สปริงยาวอย่างน้อย "
                           f"{design['smax'] / 0.5:.0f} มม. หรือเปลี่ยนเป็น gas spring"))
    if x["rd"] < 4 * x["t"]:
        w.append(("warning", f"รัศมีขอบดาย {x['rd']:.1f} มม. เล็กกว่า 4t อาจขาดที่มุม (แนะนำ 5–10t)"))
    if x["Re"] >= x["Rm"]:
        w.append(("error", "จุดคราก Re ต้องน้อยกว่า Rm"))
    if ranges:
        out = [k for k in FEATURES if not (ranges[k][0] * 0.95 <= x[k] <= ranges[k][1] * 1.05)]
        if out:
            w.append(("warning", "ค่า " + ", ".join(out) + " อยู่นอกช่วงข้อมูลที่ใช้เทรน ผลทำนายอาจคลาดเคลื่อนมาก"))
    return w
