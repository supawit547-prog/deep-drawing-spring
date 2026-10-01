"""
โปรแกรมทำนายค่าสปริงแม่พิมพ์ลากขึ้นรูป (Machine Learning)

วิธีรัน:
    pip install -r requirements.txt
    streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import dd_core as core

HERE = Path(__file__).resolve().parent
REAL_PATH = HERE / "real_data.csv"
MODEL_PATH = HERE / "model.joblib"

# โหมดออนไลน์ (Streamlit Community Cloud หรือตั้ง DD_CLOUD=1):
# ข้อมูลจริงของผู้ใช้แต่ละคนเก็บแยกในเซสชันของตัวเอง ไม่ปนกัน และไม่บันทึกลงเซิร์ฟเวอร์
CLOUD = os.environ.get("DD_CLOUD") == "1" or HERE.as_posix().startswith("/mount/src")


def get_real() -> pd.DataFrame:
    if CLOUD:
        if "real_df" not in st.session_state:
            st.session_state["real_df"] = pd.DataFrame(columns=core.REAL_COLUMNS)
        return st.session_state["real_df"].copy()
    return core.load_real(REAL_PATH)


def put_real(df: pd.DataFrame) -> None:
    if CLOUD:
        st.session_state["real_df"] = df[core.REAL_COLUMNS].reset_index(drop=True)
    else:
        core.save_real(df, REAL_PATH)


@st.cache_resource(show_spinner=False)
def default_bundle() -> dict:
    """โมเดลเริ่มต้น (ข้อมูลจำลองอย่างเดียว) ใช้ร่วมกันทุกผู้ใช้ เทรนครั้งเดียวต่อเซิร์ฟเวอร์"""
    b = core.train(core.generate_synthetic(3000, 0), pd.DataFrame(columns=core.REAL_COLUMNS), mode="mix", kind="mlp")
    b["trained_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    b["n_synth"] = 3000
    return b

st.set_page_config(page_title="ทำนายค่าสปริงแม่พิมพ์ลากขึ้นรูป", page_icon="🔩", layout="wide")


# ---------------------------------------------------------------------------
# ข้อมูลและโมเดล
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_synth(n: int, seed: int) -> pd.DataFrame:
    return core.generate_synthetic(n, seed)


def train_and_store(mode: str, kind: str, n_synth: int) -> None:
    real = get_real()
    with st.spinner("กำลังเทรนโมเดล…"):
        bundle = core.train(get_synth(n_synth, 0), real, mode=mode, kind=kind)
    bundle["trained_at"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    bundle["n_synth"] = n_synth if mode == "mix" else 0
    st.session_state["bundle"] = bundle
    if CLOUD:
        return
    try:
        joblib.dump(bundle, MODEL_PATH)
    except Exception as e:  # เช่น โฟลเดอร์เขียนไม่ได้
        st.warning(f"บันทึกไฟล์โมเดลไม่สำเร็จ: {e}")


if "bundle" not in st.session_state and CLOUD:
    with st.spinner("กำลังเตรียมโมเดล…"):
        st.session_state["bundle"] = default_bundle()
if "bundle" not in st.session_state:
    loaded = None
    if MODEL_PATH.exists():
        try:
            loaded = joblib.load(MODEL_PATH)
        except Exception:
            loaded = None
    if loaded is not None:
        st.session_state["bundle"] = loaded
    else:
        train_and_store("mix", "mlp", 3000)

if st.session_state["bundle"].get("physics_version") != core.PHYSICS_VERSION:
    # โมเดลเก่าสร้างจากสูตรชุดเดิม → เทรนใหม่ด้วยสูตรปัจจุบัน
    if CLOUD:
        st.session_state["bundle"] = default_bundle()
    else:
        train_and_store("mix", "mlp", 3000)
bundle = st.session_state["bundle"]

# ---------------------------------------------------------------------------
# แถบซ้าย: อินพุต
# ---------------------------------------------------------------------------
sb = st.sidebar

# ---------------------------------------------------------------------------
# บันทึก / เปิดงาน — ค่าในแถบซ้ายเก็บใน session_state ตาม key ของแต่ละช่อง
# ---------------------------------------------------------------------------
DEFAULTS = {"mat": "SPCC", "mu": 0.10, "t": 1.0, "dp": 60.0, "h": 25.0, "rd": 6.0, "rp": 5.0,
            "auto_d0": False, "D0": 110.0, "N": 4, "s0": 15.0, "margin": 3.0, "L0": 150.0, "sf": 1.15, "std": "iso",
            "C": 3.0, "eta": 0.5}
NUM_KEYS = ["mu", "t", "dp", "h", "rd", "rp", "D0", "s0", "margin", "L0", "sf", "C", "eta"]
LIMITS = {"mu": (0.01, 0.5), "t": (0.1, None), "dp": (1.0, None), "h": (0.5, None), "rd": (0.1, None),
          "rp": (0.0, None), "D0": (1.0, None), "s0": (0.5, None), "margin": (0.0, None), "L0": (5.0, None),
          "sf": (1.0, None), "Rm": (1.0, None), "Re": (1.0, None), "r": (0.1, None),
          "C": (1.5, 3.5), "eta": (0.4, 0.8)}


def _clamp(k: str, v) -> float:
    lo, hi = LIMITS.get(k, (None, None))
    v = float(v)
    if lo is not None:
        v = max(v, lo)
    if hi is not None:
        v = min(v, hi)
    return v
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
st.session_state.setdefault("uploader_n", 0)


def apply_job(data: dict) -> str:
    """นำค่าจากไฟล์งานเข้าแถบซ้าย คืนชื่องาน"""
    inp = data.get("inputs", data)
    mat_in = str(inp.get("mat", "")).strip()
    if mat_in not in core.MATERIALS:
        raise ValueError(f"ไม่รู้จักวัสดุ '{mat_in}'")
    st.session_state["mat"] = mat_in
    for k in ("Rm", "Re", "r"):
        if k in inp:
            st.session_state[f"{k}_{mat_in}"] = _clamp(k, inp[k])
    for k in NUM_KEYS:
        if k in inp:
            st.session_state[k] = _clamp(k, inp[k])
    if "N" in inp:
        st.session_state["N"] = max(1, int(inp["N"]))
    if inp.get("std") in ("iso", "jis"):
        st.session_state["std"] = inp["std"]
    st.session_state["auto_d0"] = bool(inp.get("auto_d0", False))
    st.session_state["job_name"] = str(data.get("name", ""))
    return str(data.get("name", "")) or "ไม่มีชื่อ"


with sb.expander("📂 เปิดงานที่บันทึกไว้", expanded=False):
    up_job = st.file_uploader("เลือกไฟล์งาน (.json)", type=["json"], key=f"job_file_{st.session_state['uploader_n']}")
    if up_job is not None:
        try:
            name_loaded = apply_job(json.loads(up_job.getvalue().decode("utf-8-sig")))
            st.session_state["job_msg"] = f"เปิดงาน '{name_loaded}' แล้ว"
        except Exception as e:
            st.session_state["job_msg"] = f"เปิดไฟล์ไม่สำเร็จ: {e}"
        st.session_state["uploader_n"] += 1      # ล้างช่องอัปโหลด กันโหลดซ้ำ
        st.rerun()
if "job_msg" in st.session_state:
    msg = st.session_state.pop("job_msg")
    (sb.error if msg.startswith("เปิดไฟล์ไม่สำเร็จ") else sb.success)(msg)

sb.header("วัสดุแผ่น")
mat_key = sb.selectbox("ชนิดวัสดุ", list(core.MATERIALS), format_func=lambda k: core.MATERIALS[k]["name"], key="mat")
mat = core.MATERIALS[mat_key]
for _k in ("Rm", "Re", "r"):
    st.session_state.setdefault(f"{_k}_{mat_key}", float(mat[_k]))
c1, c2 = sb.columns(2)
Rm = c1.number_input("Rm (MPa)", min_value=1.0, step=5.0, key=f"Rm_{mat_key}",
                     help="ความต้านแรงดึงสูงสุด Rm = Fmax / A0 จากใบรับรองวัสดุ (mill certificate) หรือผลทดสอบแรงดึง")
Re = c2.number_input("Re (MPa)", min_value=1.0, step=5.0, key=f"Re_{mat_key}",
                     help="จุดคราก Re (หรือ Rp0.2) = F_yield / A0 จากใบรับรองวัสดุ ต้องน้อยกว่า Rm")
r_val = c1.number_input("ค่า r (Lankford)", min_value=0.1, step=0.05, key=f"r_{mat_key}",
                        help="r = ln(w0/w) / ln(t0/t) จากการดึงชิ้นทดสอบ · ค่าเฉลี่ย r̄ = (r0 + 2·r45 + r90)/4 · r สูง = ลากลึกได้ดี")
mu = c2.number_input("แรงเสียดทาน μ", min_value=0.01, max_value=0.5, step=0.01, key="mu",
                     help="น้ำมันลากขึ้นรูปอย่างดี/ฟิล์ม 0.03–0.06 · น้ำมันทั่วไป 0.08–0.12 · แห้ง/ไม่หล่อลื่น 0.15–0.20")

sb.header("ขนาดชิ้นงาน (มม.)")
c1, c2 = sb.columns(2)
t = c1.number_input("ความหนา t", min_value=0.1, step=0.1, key="t", help="ความหนาแผ่นตามแบบ (nominal)")
dp = c2.number_input("พันช์ dp", min_value=1.0, step=1.0, key="dp",
                     help="= เส้นผ่านศูนย์กลางใน ของชิ้นงาน · ถ้าแบบให้ขนาดนอก: dp = d_นอก − 2t")
h = c1.number_input("ความลึก h", min_value=0.5, step=1.0, key="h",
                    help="ระยะที่พันช์ลงไปในดาย ≈ ความสูงถ้วย (วัดด้านใน) ไม่รวมขอบตัดแต่ง")
rd = c2.number_input("รัศมีดาย rd", min_value=0.1, step=0.5, key="rd",
                     help="รัศมีขอบปากดาย · (5.10) R_D = 0.035·[50 + (d0 − d1)]·√T หรือ (5.11) R_D = (5–10)·T · เล็กไป = ขาด, ใหญ่ไป = ย่น")
rp = c1.number_input("รัศมีพันช์ rp", min_value=0.0, step=0.5, key="rp",
                     help="= รัศมีมุมในก้นถ้วยตามแบบ · (5.12) r_p = (3–5)·r_d · ใช้คำนวณขนาดแผ่นเปล่า (5.13)")
auto_d0 = sb.checkbox("คำนวณ D0 จาก dp และ h อัตโนมัติ", key="auto_d0")
if auto_d0:
    D0 = round(core.blank_diameter(dp, h, rp, t), 1)
    sb.markdown(f"**D0 = {D0:.1f} มม.**")
else:
    D0 = sb.number_input("แผ่นเปล่า D0", min_value=1.0, step=1.0, key="D0",
                         help="(5.13) d0 = √(d1² + 4·(1.57·r·d1 + 2r² + h·d2)) · d1 = dp − 2rp, d2 = dp, h = ความสูงผนังตรง")

sb.header("ค่าคงที่ในสูตร")
c1, c2 = sb.columns(2)
C_const = c1.number_input("ค่าคงที่ C", min_value=1.5, max_value=3.5, step=0.1, key="C",
                          help="สมการ (5.5) แรงดันแผ่นจับชิ้นงาน · C มีค่าระหว่าง 2–3")
eta = c2.number_input("ประสิทธิภาพ η_def", min_value=0.4, max_value=0.8, step=0.05, key="eta",
                      help="สมการ (5.14) แรงสูงสุดในการลาก · η_def มีค่าระหว่าง 0.5–0.7 (ค่าน้อย = แรงมาก = ปลอดภัยกว่า)")

sb.header("การจัดวางสปริง")
c1, c2 = sb.columns(2)
N = int(c1.number_input("จำนวนสปริง", min_value=1, step=1, key="N",
                        help="ใช้จำนวนคู่ วางสมมาตรรอบแผ่นกดยึด (4, 6, 8…) · N ≥ F0 / แรงที่สปริง 1 ตัวให้ได้"))
s0 = c2.number_input("ระยะอัดล่วงหน้า (มม.)", min_value=0.5, step=0.5, key="s0",
                     help="ระยะที่สปริงถูกอัดไว้ตอนประกอบแม่พิมพ์ · ค่าน้อยสุด s0 = (h + ระยะเผื่อ)/(R − 1) โดย R = อัตราแรงท้าย/เริ่มที่ยอมรับ (≈ 2–3)")
margin = c1.number_input("ระยะเผื่อจังหวะ (มม.)", min_value=0.0, step=0.5, key="margin",
                         help="ระยะยุบเพิ่มเกิน h เผื่อการตั้งเครื่อง/สึกหรอ · ทั่วไป 2–5 มม.")
L0 = c2.number_input("ความยาวอิสระ L0 (มม.)", min_value=5.0, step=5.0, key="L0",
                     help="ความยาวสปริงตอนไม่ถูกอัด · L0 ≥ s_max / ขีดจำกัดระยะยุบ ของคลาสที่เลือก")
sf = c1.number_input("ตัวคูณเผื่อแรง", min_value=1.0, step=0.05, key="sf",
                     help="เผื่อความแปรปรวนวัสดุ/สปริงล้า · ทั่วไป 1.1–1.3")
std = c2.selectbox("มาตรฐานสี", ["iso", "jis"], format_func=lambda s: "ISO 10243" if s == "iso" else "JIS / MISUMI",
                   key="std",
                   help="ISO: เขียว=เบา น้ำเงิน=กลาง แดง=หนัก เหลือง=หนักพิเศษ · JIS: เหลือง=เบา น้ำเงิน=กลาง แดง=หนัก เขียว=หนักพิเศษ")

x = {"t": t, "D0": D0, "dp": dp, "rd": rd, "h": h, "Rm": Rm, "Re": Re, "r": r_val, "mu": mu}

# ---- บันทึกงาน ----
sb.header("💾 บันทึกงาน")
job_name = sb.text_input("ชื่องาน", key="job_name", placeholder="เช่น ถ้วย Ø60 ลึก 25 SPCC")
_fbh_s, _fd_s = core.predict(bundle, x)
_des_s = core.spring_design(_fbh_s, N, s0, h, margin, L0, sf)
job_data = {
    "app": "deep-drawing-spring", "version": 1, "name": job_name,
    "saved_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "inputs": {"mat": mat_key, "Rm": Rm, "Re": Re, "r": r_val, "mu": mu, "t": t, "dp": dp, "h": h, "rd": rd,
               "rp": rp, "auto_d0": auto_d0, "D0": D0, "N": N, "s0": s0, "margin": margin, "L0": L0, "sf": sf,
               "std": std, "C": C_const, "eta": eta},
    "results": {"F_BH_kN": round(_fbh_s, 3), "spring_k_N_per_mm": round(_des_s["k"], 1),
                "F_draw_kN": round(_fd_s, 2) if _fd_s else None, "spring_class": _des_s["best"],
                "note": "ผลลัพธ์ ณ เวลาบันทึก ใช้อ้างอิงเท่านั้น เปิดงานแล้วโปรแกรมจะคำนวณใหม่"},
}
_safe = "".join(ch for ch in (job_name or "งาน") if ch not in '\\/:*?"<>|').strip() or "งาน"
sb.download_button("💾 บันทึกงานเป็นไฟล์", json.dumps(job_data, ensure_ascii=False, indent=2).encode("utf-8"),
                   file_name=f"{_safe}.json", mime="application/json", use_container_width=True)
sb.caption("ไฟล์จะอยู่ในโฟลเดอร์ Downloads · ครั้งหน้ากด '📂 เปิดงานที่บันทึกไว้' ด้านบนสุด")

# ---------------------------------------------------------------------------
# หน้าหลัก
# ---------------------------------------------------------------------------
st.title("ทำนายค่าสปริงแม่พิมพ์ลากขึ้นรูป")
st.caption(f"โมเดล: {core.MODEL_KINDS[bundle['kind']]} · เทรนเมื่อ {bundle.get('trained_at', '-')} · "
           f"ข้อมูลจริงที่ใช้ {bundle['n_real']} แถว")

tab_pred, tab_calc, tab_model, tab_data = st.tabs(
    ["ทำนายและออกแบบสปริง", "สูตรการคำนวณ", "โมเดล ML", "ข้อมูลจริงจากโรงงาน"])


# ---------------- แท็บ 1: ทำนาย ----------------
with tab_pred:
    ph = core.physics(x, C_const, mat["bmax"], eta=eta)
    if float(ph["area"]) <= 0:
        st.error("แผ่นเปล่าเล็กเกินไป: D0 ต้องมากกว่า dp จึงจะมีพื้นที่ให้แผ่นจับชิ้นงานกด")
        st.stop()
    fam = core.MAT_FAMILY.get(mat_key, "steel")
    acr_lo, acr_hi = core.ACR[fam]
    fmax_f = float(ph["F_draw_kN"])
    fcr_lo = core.fcr_crack(dp, t, Rm, acr_lo) / 1000
    fcr_hi = core.fcr_crack(dp, t, Rm, acr_hi) / 1000

    fbh_ml, fd_ml = core.predict(bundle, x)
    fd = fd_ml if fd_ml is not None else float(ph["F_draw_kN"])
    des = core.spring_design(fbh_ml, N, s0, h, margin, L0, sf)
    press_ton = (fd + des["F_end_total"] / 1000) * 1.5 / 9.80665

    m1, m2, m3 = st.columns(3)
    m1.metric("แรงกดยึดชิ้นงาน (ML)", f"{fbh_ml:.2f} kN",
              delta=f"{(fbh_ml / float(ph['F_BH_kN']) - 1) * 100:+.0f}% เทียบสูตร (5.5)–(5.6)", delta_color="off")
    m2.metric("ค่าคงที่สปริงต่อตัว", f"{des['k']:.1f} N/มม.",
              delta=f"{N} ตัว · ตัวละ {des['F0_each'] / 1000:.2f} kN", delta_color="off")
    m3.metric("แรงลากขึ้นรูป", f"{fd:.1f} kN", delta=f"เครื่องเพรส ≈ {press_ton:.1f} ตัน", delta_color="off")
    st.caption(f"ตามสูตร: แรงกดยึด (5.6) {float(ph['F_BH_kN']):.2f} kN · แรงลากสูงสุด (5.14) {fmax_f:.1f} kN · "
               f"แรงที่ทำให้แตก (5.7) {fcr_lo:.1f}–{fcr_hi:.1f} kN · β = {float(ph['beta']):.2f} · "
               "ขนาดเครื่อง = 1.5 × (แรงลาก + แรงกดยึดท้ายจังหวะ)")
    if fmax_f >= fcr_lo:
        (st.error if fmax_f >= fcr_hi else st.warning)(
            f"แรงลากสูงสุด Fmax = {fmax_f:.1f} kN {'มากกว่า' if fmax_f >= fcr_hi else 'ใกล้หรือเกิน'}แรงที่ทำให้แตก "
            f"Fcr = {fcr_lo:.1f}–{fcr_hi:.1f} kN (a_cr {acr_lo}–{acr_hi}) — เสี่ยงก้นถ้วยขาด ควรลด β หรือเพิ่มรัศมี")

    for level, msg in core.check_warnings(x, mat, des, bundle.get("ranges")):
        (st.error if level == "error" else st.warning)(msg)

    left, right = st.columns([1, 1.2])
    with left:
        st.subheader("รายละเอียดสปริง")
        st.table(pd.DataFrame({
            "รายการ": ["ค่าคงที่สปริงต่อตัว (ขั้นต่ำ)", f"แรงต่อตัว ที่อัดล่วงหน้า {s0:.1f} มม.", "ระยะยุบรวมสูงสุด",
                       "แรงต่อตัว ท้ายจังหวะ", "แรงกดยึดรวม เริ่ม → ท้าย", "ระยะยุบ / ความยาวอิสระ"],
            "ค่า": [f"{des['k']:.1f} N/มม.", f"{des['F0_each']:.0f} N", f"{des['smax']:.1f} มม.",
                    f"{des['F_end_each']:.0f} N", f"{des['F0_total'] / 1000:.1f} → {des['F_end_total'] / 1000:.1f} kN",
                    f"{des['defl_pct']:.1f}%"],
        }).set_index("รายการ"))

    with right:
        st.subheader("เลือกคลาสสปริง")
        status_th = {"ok": "✅ ผ่าน", "mid": "⚠️ อายุสั้นลง", "bad": "❌ ยุบเกิน"}
        cls = pd.DataFrame([{
            "สี": core.COLOR_TH[c[std]], "คลาส": c["name"] + (" ⭐ แนะนำ" if c["id"] == des["best"] else ""),
            "ยุบได้ อายุยาว/สูงสุด": f"{c['life']}% / {c['max']}%", "L0 ขั้นต่ำ (มม.)": f"{c['min_L0']:.0f}",
            "ผล": status_th[c["status"]],
        } for c in des["classes"]])
        st.dataframe(cls, hide_index=True, use_container_width=True)
        st.caption("ขีดจำกัดระยะยุบเป็นค่าประมาณ ตรวจสอบกับแคตตาล็อกผู้ผลิต แล้วเลือกขนาดที่ค่าคงที่สปริง ≥ ค่าที่ต้องการ")

    st.divider()
    st.subheader("บันทึกผลลองแม่พิมพ์จริง")
    st.caption("หลังลองพิมพ์ ใส่แรงกดยึดที่ได้ชิ้นงานดี (ไม่ย่น ไม่ขาด) ระบบจะเก็บพร้อมค่าที่กรอกในแถบซ้าย")
    with st.form("add_real", clear_on_submit=True):
        f1, f2, f3 = st.columns([1, 1, 2])
        real_fbh = f1.number_input("แรงกดยึดจริง (kN)", min_value=0.0, value=0.0, step=0.1)
        real_fd = f2.number_input("แรงลากจริง (kN) ไม่บังคับ", min_value=0.0, value=0.0, step=0.1)
        note = f3.text_input("หมายเหตุ (เช่น รหัสชิ้นงาน, ผลที่ได้)")
        if st.form_submit_button("เพิ่มเข้าชุดข้อมูล"):
            if real_fbh <= 0:
                st.error("ใส่แรงกดยึดจริงเป็นตัวเลขมากกว่า 0")
            else:
                real = get_real()
                row = {**x, "F_BH_kN": real_fbh, "F_draw_kN": real_fd if real_fd > 0 else np.nan,
                       "material": mat_key, "note": note, "date": dt.date.today().isoformat()}
                real = pd.concat([real, pd.DataFrame([row])], ignore_index=True)
                put_real(real)
                st.success(f"เพิ่มแล้ว · ข้อมูลจริงในระบบ {len(real)} แถว — ไปที่แท็บ 'โมเดล ML' เพื่อเทรนใหม่")

# ---------------- แท็บ: สูตรการคำนวณ ----------------
with tab_calc:
    st.caption("สูตรที่ใช้ทั้งหมด พร้อมแทนค่าจากที่กรอกในแถบซ้าย (หน่วย มม., MPa, N, kN) "
               "ค่าคงที่ในสูตรเชิงประสบการณ์เป็นค่าประมาณ ควรยืนยันด้วยการลองแม่พิมพ์")

    beta = float(ph["beta"])
    p_bh = float(ph["p"])
    area = float(ph["area"])
    f_siebel = float(ph["F_BH_kN"])
    base_bh, _ = core.baseline(pd.DataFrame([x]))
    corr = fbh_ml / float(base_bh[0])

    # 0. การเลือกค่าอินพุต
    st.subheader("0. การหาค่าอินพุต")
    with st.expander("วัสดุแผ่น: Rm, Re, r, μ — คืออะไร หามาจากไหน", expanded=False):
        st.info("สรุปหน้างาน: ขอใบ **Mill Certificate (Mill sheet)** จากผู้ขายวัสดุทุกล็อต แล้วนำค่า Rm, Re, r มากรอก "
                "จะแม่นกว่าค่าตั้งต้นในโปรแกรม เพราะวัสดุแต่ละล็อตต่างกันได้ ±10%")

        st.markdown("#### Rm — ความต้านแรงดึงสูงสุด (Tensile Strength)")
        st.markdown("แรงดึงสูงสุดต่อพื้นที่ที่วัสดุรับได้ก่อนขาด หน่วย MPa (= N/mm²)")
        st.latex(r"R_m=\frac{F_{max}}{A_0},\qquad A_0=b_0\times t_0")
        st.markdown("- **F_max** แรงดึงสูงสุดที่เครื่องทดสอบอ่านได้ (N)\n"
                    "- **A₀** พื้นที่หน้าตัดชิ้นทดสอบก่อนดึง = ความกว้าง b₀ × ความหนา t₀ (mm²)")
        st.latex(r"\text{ex. }b_0=25,\ t_0=0.8\Rightarrow A_0=20\ \text{mm}^2,\quad "
                 r"F_{max}=6600\ \text{N}\Rightarrow R_m=\frac{6600}{20}=330\ \text{MPa}")
        st.markdown("**หามาจาก**\n"
                    "1. **Mill sheet** ช่อง **T.S.** หรือ **Tensile Strength** — ง่ายที่สุด เป็นค่าจริงของล็อตนั้น\n"
                    "2. **ทดสอบแรงดึงเอง** ด้วยเครื่อง UTM ตาม JIS Z 2241 (หรือส่งห้องแล็บ)\n"
                    "3. **ค่าตามมาตรฐาน** เช่น SPCC (JIS G 3141) กำหนดขั้นต่ำ 270 MPa ของจริงมัก 300–350 MPa "
                    "— โปรแกรมใส่ค่าเฉลี่ยไว้ให้เมื่อเลือกวัสดุ")
        st.caption("ผล: Rm สูง → ต้องใช้แรงลากและแรงกดยึดมากขึ้นตามสัดส่วน")

        st.markdown("#### Re — จุดคราก (Yield Strength / Yield Point)")
        st.markdown("แรงต่อพื้นที่ที่วัสดุเริ่มเสียรูปถาวร ดึงเกินจุดนี้แล้วปล่อย ชิ้นงานจะไม่คืนรูปเดิม")
        st.latex(r"R_e=\frac{F_{yield}}{A_0}")
        st.markdown("- **F_yield** แรงตอนวัสดุเริ่มคราก (N)\n\n"
                    "**หามาจาก** Mill sheet ช่อง **Y.P.** (Yield Point) หรือ **Y.S.** (Yield Strength) — "
                    "วัสดุที่ไม่มีจุดครากชัด เช่น สแตนเลส อะลูมิเนียม ใช้ค่า **Rp0.2** (แรงที่ทำให้ยืดถาวร 0.2%)")
        st.latex(rf"\frac{{R_e}}{{R_m}}=\frac{{{Re:.0f}}}{{{Rm:.0f}}}={Re / Rm:.2f}")
        ratio_txt = ("ลากลึกได้ดี" if Re / Rm < 0.65 else "ปานกลาง" if Re / Rm <= 0.8 else "ย่นและเด้งกลับง่าย ระวัง")
        st.caption(f"อัตราส่วน Re/Rm < 0.65 เนื้อยืดดี ลากลึกได้ดี · > 0.8 ย่นและเด้งกลับ (spring-back) ง่าย "
                   f"— วัสดุที่กรอกตอนนี้: {ratio_txt}")

        st.markdown("#### r — ค่า Lankford (ค่าความต้านทานการบางตัว)")
        st.markdown("บอกว่าตอนแผ่นถูกดึง เนื้อวัสดุหดทาง **ความกว้าง** มากกว่าทาง **ความหนา** แค่ไหน "
                    "ค่ายิ่งสูง แผ่นยิ่งไม่บาง → ลากลึกได้ดี ไม่ขาดง่าย")
        st.latex(r"r=\frac{\varepsilon_w}{\varepsilon_t}=\frac{\ln(w_0/w)}{\ln(t_0/t)}")
        st.markdown("- **w₀, w** ความกว้างชิ้นทดสอบ ก่อน/หลัง ดึง\n"
                    "- **t₀, t** ความหนา ก่อน/หลัง ดึง\n"
                    "- **ε_w, ε_t** ความเครียด (strain) ตามกว้าง / ตามหนา\n\n"
                    "แผ่นรีดมีทิศทาง จึงวัด 3 ทิศเทียบแนวรีด (0°, 45°, 90°) แล้วเฉลี่ย")
        st.latex(r"\bar r=\frac{r_0+2r_{45}+r_{90}}{4}")
        st.markdown("**หามาจาก**\n"
                    "1. Mill sheet ของเหล็กเกรดลากลึก (SPCE, SPCF, DC04 ฯลฯ) มักระบุ **r-value**\n"
                    "2. ทดสอบตาม ISO 10113 (ห้องแล็บ)\n"
                    "3. ไม่มีข้อมูล → ใช้ค่าตั้งต้นในโปรแกรมได้ ผลคลาดเคลื่อนไม่มาก")
        st.caption(f"ค่าที่ใช้ r = {r_val:.2f} · เหล็กลากลึก 1.6–2.0 · SPCC 1.2–1.5 · สแตนเลส/อะลูมิเนียม 0.6–1.0")

        st.markdown("#### μ — สัมประสิทธิ์แรงเสียดทาน")
        st.markdown("บอกว่าผิวแผ่นลื่นแค่ไหนเมื่อไถลผ่านขอบดายและใต้แผ่นกดยึด วัดเองได้ยาก "
                    "ในทางปฏิบัติ **เลือกตามการหล่อลื่นที่ใช้จริง**")
        st.table(pd.DataFrame({"การหล่อลื่น": ["น้ำมันลากขึ้นรูปอย่างดี / ฟิล์มพลาสติก", "น้ำมันทั่วไป", "แห้ง / ไม่หล่อลื่น"],
                               "μ": ["0.03–0.06", "0.08–0.12", "0.15–0.20"]}).set_index("การหล่อลื่น"))

    with st.expander("ขนาดชิ้นงาน: t, dp, h, rd, rp, D0 — คืออะไร หามาจากไหน", expanded=False):
        st.markdown("ขนาดทั้งหมดหาได้จาก **แบบชิ้นงาน (drawing)** ส่วนรัศมีดายและพันช์เลือกตอนออกแบบแม่พิมพ์")
        st.table(pd.DataFrame({
            "สัญลักษณ์": ["t", "dp", "h", "rp", "rd", "D0", "d_out, H_out", "D0,trim"],
            "ความหมาย": ["ความหนาแผ่น", "เส้นผ่านศูนย์กลางพันช์ = ขนาดในของถ้วย", "ความลึกการลาก ≈ ความสูงในของถ้วย",
                         "รัศมีมุมพันช์ = รัศมีมุมในก้นถ้วย", "รัศมีขอบปากดาย", "เส้นผ่านศูนย์กลางแผ่นเปล่าก่อนลาก",
                         "ขนาดนอก / ความสูงนอก ตามแบบ", "แผ่นเปล่าที่เผื่อขอบไว้ตัดแต่งหลังลาก"],
            "หามาจาก": ["แบบชิ้นงาน / Mill sheet", "แบบชิ้นงาน (ถ้าให้ขนาดนอก ลบ 2t)", "แบบชิ้นงาน (ถ้าให้ความสูงนอก ลบ t)",
                        "แบบชิ้นงาน", "ออกแบบแม่พิมพ์ 5–10t", "คำนวณ (หัวข้อที่ 1)", "แบบชิ้นงาน", "D0 × 1.02–1.05"],
        }).set_index("สัญลักษณ์"))
        st.markdown("**เส้นผ่านศูนย์กลางพันช์** — ถ้าแบบให้ขนาดนอก")
        st.latex(rf"d_p=d_{{out}}-2t\qquad(\text{{ex. }}d_{{out}}={dp + 2 * t:.1f}\Rightarrow d_p={dp:.1f})")
        st.markdown("**ความลึกการลาก** — ถ้าแบบให้ความสูงนอก")
        st.latex(rf"h=H_{{out}}-t\qquad(\text{{ex. }}H_{{out}}={h + t:.1f}\Rightarrow h={h:.1f})")
        st.markdown("**รัศมีพันช์และรัศมีดาย** — rd เล็กไป = ชิ้นงานขาดที่มุม, ใหญ่ไป = ย่นที่ปากถ้วย")
        st.latex(rf"r_d=(5\text{{–}}10)\,T={5 * t:.1f}\text{{–}}{10 * t:.1f}\ \text{{mm}},\qquad "
                 rf"r_p=(3\text{{–}}5)\,r_d={3 * rd:.1f}\text{{–}}{5 * rd:.1f}\ \text{{mm}}")
        st.markdown("**แผ่นเปล่ารวมขอบตัดแต่ง** — ขอบถ้วยหลังลากมักไม่เรียบ (earing) ต้องเผื่อไว้ตัด")
        D0_trim = core.blank_diameter(dp, h, rp, t)
        st.latex(rf"D_{{0,trim}}=D_0\times(1.02\text{{–}}1.05)={D0_trim * 1.02:.1f}\text{{–}}{D0_trim * 1.05:.1f}\ \text{{mm}}")
        st.caption("รายละเอียดสูตร D0 อยู่ในหัวข้อที่ 1")

    with st.expander("การจัดวางสปริง: N, s0, ระยะเผื่อ, L0, ตัวคูณเผื่อ", expanded=True):
        R_target = 2.5
        s0_min = (h + margin) / (R_target - 1)
        st.markdown("**ระยะอัดล่วงหน้าน้อยสุด** เพื่อให้แรงท้ายจังหวะไม่เกิน R เท่าของแรงเริ่ม")
        st.latex(r"\frac{F_{end}}{F_0}=\frac{s_0+h+s_{margin}}{s_0}\le R\;\Rightarrow\;s_0\ge\frac{h+s_{margin}}{R-1}")
        st.latex(rf"s_{{0,min}}=\frac{{{h:.1f}+{margin:.1f}}}{{{R_target}-1}}={s0_min:.1f}\ \text{{mm}}\quad(R={R_target})")
        st.caption(f"ค่าที่กรอก s0 = {s0:.1f} มม. → R = {des['ratio']:.2f} · R ≈ 2–3 ใช้ได้สำหรับสปริงขด "
                   "ถ้าต้องการแรงเกือบคงที่ (R ≈ 1.1–1.3) ใช้ gas spring")
        st.markdown("**ความยาวอิสระน้อยสุด** ตามคลาสสปริง")
        st.latex(r"L_{0,min}=\frac{s_{max}}{\delta_{allow}}")
        best_c = next((c for c in core.CLASSES if c["id"] == des["best"]), core.CLASSES[1])
        st.latex(rf"L_{{0,min}}=\frac{{{des['smax']:.1f}}}{{{best_c['life'] / 100:.2f}}}={des['smax'] / (best_c['life'] / 100):.0f}\ \text{{mm}}"
                 rf"\quad(\delta_{{allow}}={best_c['life']}\%)")
        st.caption("δ_allow: เบา 30% · กลาง 25% · หนัก 20% · หนักพิเศษ 17% (อายุยาว ≥ 10⁶ ครั้ง, ค่าประมาณ)")
        st.markdown("**จำนวนสปริง** จากแรงที่สปริง 1 ตัวในแคตตาล็อกให้ได้ที่ระยะ s0")
        st.latex(r"N\ge\frac{F_0}{k_{cat}\cdot s_0}\quad\text{(round up to even)}")
        st.latex(rf"\text{{ex. }}k_{{cat}}=100\ \text{{N/mm}}\Rightarrow N\ge\frac{{{des['F0_total']:.0f}}}{{100\times{s0:.1f}}}"
                 rf"={des['F0_total'] / (100 * s0):.1f}\Rightarrow N={max(2, 2 * math.ceil(des['F0_total'] / (100 * s0) / 2))}")
        st.markdown("**ตัวคูณเผื่อแรง** SF ≈ 1.1–1.3 เผื่อความแปรปรวนวัสดุและสปริงล้าเมื่อใช้งานนาน")

    # 1. ขนาดแผ่นเปล่า
    st.subheader("1. ขนาดแผ่นเปล่าและขั้นตอนการลาก")
    d1b = max(dp - 2 * rp, 0.0)
    hw = max(h - rp, 0.0)
    D0_calc = core.blank_diameter(dp, h, rp, t)
    st.markdown("**ขนาดแผ่นโลหะเปล่า** สมการ (5.13)")
    st.latex(r"d_0=\sqrt{d_1^2+4\,(1.57\,r\,d_1+2r^2+h\,d_2)}")
    st.latex(rf"d_1=d_p-2r={dp:.1f}-2({rp:.1f})={d1b:.1f},\quad d_2=d_p={dp:.1f},\quad h=h_{{total}}-r={h:.1f}-{rp:.1f}={hw:.1f}")
    st.latex(rf"d_0=\sqrt{{{d1b:.1f}^2+4\,[1.57({rp:.1f})({d1b:.1f})+2({rp:.1f})^2+({hw:.1f})({dp:.1f})]}}={D0_calc:.2f}\ \text{{mm}}")
    st.caption(f"ค่า D0 ที่ใช้คำนวณตอนนี้ = {D0:.1f} มม. · d1 = เส้นผ่านศูนย์กลางก้นเรียบ, d2 = เส้นผ่านศูนย์กลางถ้วย, "
               "r = รัศมีพั้นช์, h = ความสูงผนังตรง")

    st.markdown("**อัตราส่วนการลากขึ้นรูป** สมการ (5.8)")
    st.latex(rf"\beta=\frac{{d_0}}{{d_1}}=\frac{{{D0:.1f}}}{{{dp:.1f}}}={beta:.2f}\qquad \beta_{{max}}={mat['bmax']}")
    stages = core.draw_stages(D0, dp, mat_key)
    bnext = core.MAT_EXTRA.get(mat_key, (0.07, 1.25))[1]
    if len(stages) == 1:
        st.success(f"β = {beta:.2f} ≤ βmax = {mat['bmax']} → ลากขึ้นรูปได้ในขั้นตอนเดียว")
    else:
        st.warning(f"β = {beta:.2f} > βmax = {mat['bmax']} → ต้องลากประมาณ {len(stages)} ครั้ง: "
                   + " → ".join(f"Ø{d:.1f}" for d in stages[:-1]) + f" → Ø{dp:.1f} มม.")
        st.caption(f"ลากซ้ำใช้ β ≈ {bnext} (ไม่อบอ่อน) หรือ 1.7 (หลังอบอ่อน) ตามตารางที่ 5.2")

    # 2. แม่พิมพ์
    st.subheader("2. ขนาดแม่พิมพ์")
    kc = core.MAT_EXTRA.get(mat_key, (0.07, 1.25))[0]
    clr = core.die_clearance(t, mat_key)
    rd_oe = core.die_radius_recommend(D0, dp, t)
    st.markdown("**ระยะช่องว่างแม่พิมพ์** สมการ (5.9)")
    st.latex(rf"CL=T+{kc}\sqrt{{10T}}={t:.2f}+{kc}\sqrt{{10({t:.2f})}}={clr:.3f}\ \text{{mm}}")
    st.latex(rf"D_{{die}}=d_p+2\,CL={dp:.2f}+2({clr:.3f})={dp + 2 * clr:.2f}\ \text{{mm}}")
    st.caption("สมการ (5.9) สำหรับแผ่นเหล็ก (k = 0.07) · ถ้าใช้ช่องว่างน้อยกว่าความหนา ผนังถ้วยจะถูกรีด (ironing) ต้องใช้แรงเพิ่ม")
    st.markdown("**รัศมีดาย** สมการ (5.10) Oehler & Kaiser และ (5.11) Sellin")
    st.latex(rf"R_D=0.035\,[50+(d_0-d_1)]\sqrt{{T}}=0.035\,[50+({D0:.1f}-{dp:.1f})]\sqrt{{{t:.2f}}}={rd_oe:.2f}\ \text{{mm}}")
    st.latex(rf"R_D=(5\text{{–}}10)\,T={5 * t:.1f}\text{{–}}{10 * t:.1f}\ \text{{mm}}")
    st.markdown("**รัศมีพั้นช์** สมการ (5.12)")
    st.latex(rf"r_p=(3\text{{–}}5)\,r_d=(3\text{{–}}5)({rd:.1f})={3 * rd:.1f}\text{{–}}{5 * rd:.1f}\ \text{{mm}}")
    st.caption(f"ค่าที่กรอก: r_d = {rd:.1f} มม. ({rd / t:.1f}T) · r_p = {rp:.1f} มม. ({rp / rd:.1f} เท่าของ r_d)")

    # 3. แรงกดยึด
    st.subheader("3. แรงกดบนแผ่นจับชิ้นงาน")
    st.markdown("**แรงดันในการกดแผ่นจับชิ้นงาน** สมการ (5.5)")
    st.latex(r"P_{BH}=10^{-3}\,C\left[(\beta-1)^3+\frac{0.005\,d_0}{T}\right]\sigma_u")
    st.latex(rf"P_{{BH}}=10^{{-3}}({C_const:.1f})\left[({beta:.2f}-1)^3+\frac{{0.005({D0:.1f})}}{{{t:.2f}}}\right]({Rm:.0f})"
             rf"={p_bh:.3f}\ \text{{N/mm}}^2")
    st.markdown("**พื้นที่ปีกที่แผ่นจับชิ้นงานกด และแรงกด** สมการ (5.6)")
    st.latex(rf"A_{{BH}}=\frac{{\pi}}{{4}}(d_0^2-d_1^2)=\frac{{\pi}}{{4}}({D0:.1f}^2-{dp:.1f}^2)={area:.2f}\ \text{{mm}}^2")
    st.latex(rf"F_{{BH}}=P_{{BH}}\times A_{{BH}}={p_bh:.3f}\times{area:.2f}={f_siebel * 1000:.0f}\ \text{{N}}={f_siebel:.2f}\ \text{{kN}}")
    st.markdown("**ค่าจากโมเดล ML** = ค่าสูตรพื้นฐาน × ตัวแก้ที่โมเดลเรียนรู้")
    st.latex(rf"F_{{BH,ML}}=F_{{BH,base}}\times e^{{\hat y}}={float(base_bh[0]):.2f}\times{corr:.3f}"
             rf"={fbh_ml:.2f}\ \text{{kN}}")
    st.caption("F_BH,base ใช้สมการ (5.5)–(5.6) ที่ C = 3.0 · ผลของวัสดุ ค่า r แรงเสียดทาน และข้อมูลจริง อยู่ในตัวแก้ของโมเดล")

    # 4. แรงลาก / แรงแตก
    st.subheader("4. แรงลากขึ้นรูปและแรงที่ทำให้แตก")
    dm = dp + t
    st.markdown("**แรงสูงสุดในการลากขึ้นรูป** สมการ (5.14)")
    st.latex(r"F_{max}=\pi\,d_m\,T\left[1.1\,\frac{1.3\,\sigma_u}{\eta_{def}}\left(\ln\frac{d_0}{d_1}-0.25\right)\right],\quad d_m=d_1+T")
    st.latex(rf"F_{{max}}=\pi({dm:.2f})({t:.2f})\left[1.1\,\frac{{1.3({Rm:.0f})}}{{{eta:.2f}}}"
             rf"\left(\ln\frac{{{D0:.1f}}}{{{dp:.1f}}}-0.25\right)\right]={fmax_f * 1000:.0f}\ \text{{N}}={fmax_f:.2f}\ \text{{kN}}")
    st.caption(f"ค่าจากโมเดล ML = {fd:.1f} kN · η_def น้อย → แรงมาก (ปลอดภัยกว่า) · ช่วง η 0.5–0.7 ให้ "
               f"{float(core.fmax_draw(dp, t, D0, Rm, 0.7)) / 1000:.1f}–{float(core.fmax_draw(dp, t, D0, Rm, 0.5)) / 1000:.1f} kN")
    st.markdown("**แรงที่ทำให้เกิดการแตก** สมการ (5.7)")
    st.latex(r"F_{cr}=\pi\,d_m\,T\,\sigma_u\,a_{cr}")
    st.latex(rf"F_{{cr}}=\pi({dm:.2f})({t:.2f})({Rm:.0f})({acr_lo}\text{{–}}{acr_hi})={fcr_lo:.2f}\text{{–}}{fcr_hi:.2f}\ \text{{kN}}")
    st.dataframe(pd.DataFrame({"วัสดุ": ["แผ่นเหล็กทั่วไป (SAE 1006)", "เหล็กกล้าไร้สนิม (AISI 304)",
                                         "ทองเหลือง (UNS C27000)", "อะลูมิเนียม (AA 1050-0)"],
                               "a_cr": ["1.05–1.55", "0.95–1.30", "0.92–1.27", "0.99–1.22"]}),
                 hide_index=True)
    if fmax_f < fcr_lo:
        st.success(f"Fmax = {fmax_f:.2f} kN < Fcr ต่ำสุด = {fcr_lo:.2f} kN → ลากขึ้นรูปได้โดยไม่แตก (แม้ใช้ a_cr ต่ำสุด)")
    elif fmax_f < fcr_hi:
        st.warning(f"Fmax = {fmax_f:.2f} kN อยู่ระหว่าง Fcr {fcr_lo:.2f}–{fcr_hi:.2f} kN → ผ่านเฉพาะเมื่อใช้ a_cr สูง "
                   "ควรเผื่อความปลอดภัยโดยเทียบกับ a_cr ต่ำสุด")
    else:
        st.error(f"Fmax = {fmax_f:.2f} kN > Fcr สูงสุด = {fcr_hi:.2f} kN → ชิ้นงานจะแตก")

    # 5. สปริง
    st.subheader("5. สปริงแม่พิมพ์")
    st.latex(rf"F_0=F_{{BH}}\times SF={fbh_ml:.2f}\times{sf:.2f}={des['F0_total'] / 1000:.2f}\ \text{{kN}}")
    st.latex(rf"k=\frac{{F_0}}{{N\cdot s_0}}=\frac{{{des['F0_total']:.0f}}}{{{N}\times{s0:.1f}}}={des['k']:.1f}\ \text{{N/mm}}")
    st.latex(rf"s_{{max}}=s_0+h+s_{{margin}}={s0:.1f}+{h:.1f}+{margin:.1f}={des['smax']:.1f}\ \text{{mm}}")
    st.latex(rf"F_{{end}}=N\cdot k\cdot s_{{max}}={N}\times{des['k']:.1f}\times{des['smax']:.1f}"
             rf"={des['F_end_total']:.0f}\ \text{{N}}={des['F_end_total'] / 1000:.1f}\ \text{{kN}}")
    st.latex(rf"\frac{{F_{{end}}}}{{F_0}}=\frac{{s_{{max}}}}{{s_0}}={des['ratio']:.2f}"
             rf"\qquad \frac{{s_{{max}}}}{{L_0}}\times100={des['defl_pct']:.1f}\%")
    st.caption("ถ้าใช้สปริงจากแคตตาล็อก ให้คำนวณซ้ำด้วยค่า k จริงของรุ่นที่เลือก: F0 = N·k·s0 ต้องไม่น้อยกว่าแรงกดยึดที่ต้องการ")

    # 6. ดายคูชั่น
    st.subheader("6. ถ้าใช้ดายคูชั่นแทนสปริง")
    st.caption("ดายคูชั่นให้แรงกดเกือบคงที่ตลอดจังหวะ (P_BH คงที่) · แรงแปรผันตรงกับแรงดันลม")
    cc1, cc2 = st.columns(2)
    cap = cc1.number_input("ความจุดายคูชั่น (kN)", min_value=1.0, value=63.0, step=1.0)
    prt = cc2.number_input("ที่แรงดันลม (MPa)", min_value=0.05, value=0.5, step=0.05)
    p_air_f = core.cushion_air_pressure(f_siebel, cap, prt)
    p_air_ml = core.cushion_air_pressure(fbh_ml, cap, prt)
    st.latex(rf"P_{{air}}=\frac{{F_{{BH}}}}{{F_{{rated}}}}\times P_{{rated}}=\frac{{{f_siebel:.2f}}}{{{cap:.0f}}}\times{prt:.2f}"
             rf"={p_air_f:.3f}\ \text{{MPa}}\ (\approx{p_air_f * 10:.2f}\ \text{{bar}})")
    st.caption(f"จากค่า ML ({fbh_ml:.2f} kN) ต้องตั้งลม ≈ {p_air_ml:.3f} MPa · ค่าเริ่มต้นตามตารางที่ 5.3 (NNCY-8-1: 63 kN ที่ 0.5 MPa) "
               "· ถ้ากราฟของเครื่องจริงไม่ผ่านจุดศูนย์ ให้อ่านจากกราฟของเครื่องแทน")

    # 7. เครื่องเพรส
    st.subheader("7. แรงในการเลือกเครื่องจักร")
    st.latex(rf"P=\frac{{1.5\,(F_d+F_{{end}})}}{{9.807}}=\frac{{1.5\,({fd:.1f}+{des['F_end_total'] / 1000:.1f})}}{{9.807}}"
             rf"={press_ton:.1f}\ \text{{ton}}")
    st.caption("ตัวคูณ 1.5 เท่าของแรงลากขึ้นรูป · รวมแรงกดยึดท้ายจังหวะด้วย เพราะสปริง/คูชั่นดันสวนแรมตลอดจังหวะ")

# ---------------- แท็บ 2: โมเดล ----------------
with tab_model:
    real_now = get_real()
    c1, c2, c3, c4 = st.columns([1.3, 1.3, 1, 1])
    kind = c1.selectbox("ชนิดโมเดล", list(core.MODEL_KINDS), format_func=core.MODEL_KINDS.get,
                        index=list(core.MODEL_KINDS).index(bundle["kind"]))
    mode = c2.selectbox("ข้อมูลที่ใช้เทรน", ["mix", "real"],
                        format_func=lambda m: "ข้อมูลจำลอง + ข้อมูลจริง" if m == "mix" else "ข้อมูลจริงเท่านั้น (≥ 15 แถว)")
    n_synth = c3.number_input("จำนวนข้อมูลจำลอง", 500, 20000, 3000, step=500, disabled=(mode == "real"))
    c4.write("")
    c4.write("")
    if c4.button("เทรนโมเดลใหม่", type="primary", use_container_width=True):
        try:
            train_and_store(mode, kind, int(n_synth))
            st.rerun()
        except ValueError as e:
            st.error(str(e))

    st.caption(f"ข้อมูลจริงในไฟล์ตอนนี้ {len(real_now)} แถว · โมเดลปัจจุบันเทรนด้วย {bundle['n_train']} แถว "
               f"ทดสอบ {bundle['n_test']} แถว")

    mt = bundle["metrics"]
    cols = st.columns(5)
    cols[0].metric("R² แรงกดยึด", f"{mt['r2_bh']:.3f}")
    cols[1].metric("คลาดเคลื่อนเฉลี่ย แรงกดยึด", f"{mt['mape_bh']:.1f}%")
    if "r2_fd" in mt:
        cols[2].metric("R² แรงลาก", f"{mt['r2_fd']:.3f}")
        cols[3].metric("คลาดเคลื่อนเฉลี่ย แรงลาก", f"{mt['mape_fd']:.1f}%")
    if "mape_bh_real" in mt:
        cols[4].metric("คลาดเคลื่อน เฉพาะข้อมูลจริง", f"{mt['mape_bh_real']:.1f}%",
                       delta=f"{mt['n_test_real']} แถวทดสอบ", delta_color="off")

    g1, g2 = st.columns(2)
    with g1:
        st.markdown("**ค่าทำนายเทียบค่าจริง (ชุดทดสอบ)**")
        te = bundle["test"]
        fig, ax = plt.subplots(figsize=(5, 4.2))
        syn = te[~te["is_real"]]
        rl = te[te["is_real"]]
        ax.scatter(syn["F_BH_kN"], syn["pred_F_BH_kN"], s=8, alpha=0.4, color="#0D5A6D", label="synthetic")
        if len(rl):
            ax.scatter(rl["F_BH_kN"], rl["pred_F_BH_kN"], s=30, color="#C0561A", label="real (factory)")
        lo = min(te["F_BH_kN"].min(), te["pred_F_BH_kN"].min()) * 0.8
        hi = max(te["F_BH_kN"].max(), te["pred_F_BH_kN"].max()) * 1.2
        ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Actual blank holder force (kN)")
        ax.set_ylabel("Predicted (kN)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with g2:
        st.markdown("**ความสำคัญของตัวแปร (permutation importance)**")
        imp = bundle["importance"]
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ax.barh(imp.index, imp.values, color="#0D5A6D")
        ax.set_xlabel("Increase in error when shuffled")
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("แท่งยาว = ตัวแปรนั้นมีผลต่อแรงกดยึดมาก")

    with st.expander("วิธีการทำงานของโมเดล"):
        st.markdown(
            "- โมเดลเรียนรู้ 'ค่าแก้' เทียบกับสูตร Siebel (physics-informed) จึงเทรนด้วยข้อมูลจริงไม่กี่สิบแถวได้\n"
            "- อินพุต 9 ตัว: t, D0, dp, rd, h, Rm, Re, r, μ (7 ตัวแรกแปลงเป็น log แล้ว standardize)\n"
            "- เอาต์พุต: log(ค่าจริง / ค่าสูตร) ของแรงกดยึดและแรงลาก (แยกเป็น 2 โมเดล)\n"
            "- ข้อมูลเริ่มต้นเป็นข้อมูลจำลองจากสูตร Siebel  p = c·10⁻³·[(β−1)² + D0/(200t)]·Rm "
            "บวกผลของค่า r, อัตราส่วน Re/Rm, rd/t และสัญญาณรบกวนสุ่ม — ก่อนมีข้อมูลจริง โมเดลจึงให้ผลใกล้เคียงสูตรตำรา\n"
            "- ในโหมดผสม ข้อมูลจริงถูกทำซ้ำ 3–20 เท่าเพื่อเพิ่มน้ำหนัก และแบ่งชุดทดสอบตามแถวต้นฉบับ (ไม่รั่วข้ามชุด)\n"
            "- ค่าคงที่สปริง k = (F_BH × ตัวคูณเผื่อ) / (N × ระยะอัดล่วงหน้า)"
        )

# ---------------- แท็บ 3: ข้อมูลจริง ----------------
with tab_data:
    st.markdown("คอลัมน์ที่ต้องมี: `t, D0, dp, rd, h, Rm, Re, r, mu, F_BH_kN` · ไม่บังคับ: `F_draw_kN, material, note, date` "
                "(หน่วย มม., MPa, kN)")
    up = st.file_uploader("นำเข้าไฟล์ CSV หรือ Excel", type=["csv", "xlsx", "xls"])
    if up is not None and st.button("เพิ่มข้อมูลจากไฟล์"):
        try:
            raw = pd.read_excel(up) if up.name.lower().endswith(("xlsx", "xls")) else pd.read_csv(up, sep=None, engine="python")
            new, skipped = core.clean_real(raw)
            real = pd.concat([get_real(), new], ignore_index=True)
            put_real(real)
            st.success(f"นำเข้า {len(new)} แถว" + (f" (ข้าม {skipped} แถวที่ข้อมูลไม่ครบ)" if skipped else "")
                       + f" · รวม {len(real)} แถว")
        except Exception as e:
            st.error(f"อ่านไฟล์ไม่สำเร็จ: {e}")

    real = get_real()
    st.markdown(f"**ข้อมูลจริงในระบบ: {len(real)} แถว** (แก้ไข เพิ่ม หรือลบแถวในตารางได้ แล้วกดบันทึก)")
    edited = st.data_editor(real, num_rows="dynamic", use_container_width=True, key="editor")
    b1, b2, b3 = st.columns(3)
    if b1.button("บันทึกการแก้ไข"):
        try:
            cleaned, skipped = core.clean_real(edited)
            put_real(cleaned)
            st.success(f"บันทึก {len(cleaned)} แถว" + (f" (ตัด {skipped} แถวที่ข้อมูลไม่ครบ)" if skipped else ""))
        except ValueError as e:
            st.error(str(e))
    b2.download_button("ดาวน์โหลดข้อมูลจริง (CSV)", real.to_csv(index=False).encode("utf-8-sig"),
                       "real_data.csv", "text/csv")
    template = pd.DataFrame(columns=core.REAL_COLUMNS)
    b3.download_button("ดาวน์โหลดแม่แบบ (CSV)", template.to_csv(index=False).encode("utf-8-sig"),
                       "real_data_template.csv", "text/csv")
    if CLOUD:
        st.info("เวอร์ชันออนไลน์: ข้อมูลที่เพิ่มจะอยู่เฉพาะในหน้านี้ของคุณ และหายเมื่อปิดหรือรีเฟรชหน้า "
                "กด 'ดาวน์โหลดข้อมูลจริง' เก็บไว้ก่อนปิด แล้วครั้งหน้านำเข้าไฟล์เดิมได้")
    else:
        st.caption(f"ไฟล์ข้อมูลเก็บที่ {REAL_PATH}")
