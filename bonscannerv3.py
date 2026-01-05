# scannerv2_fixed.py
# Streamlit Web-App: Kassenzettel scannen -> OpenAI Vision extrahiert -> JSON -> SQLite speichern -> Auswertungen
# Features:
# - User+Passwort Login (PBKDF2-Hash, keine Klartext-PWs)
# - Pro User eigene SQLite-DB (getrennte Daten)
# - Progressbar/Status während Extraktion
# - Vision direkt an OpenAI (openai>=1.0.0 API)
# - Tages-/Wochen-/Monats-Übersichten + einfache Charts
#
# Deployment:
# - Streamlit Cloud: Secrets setzen: OPENAI_API_KEY
# - Lokal: ENV setzen: OPENAI_API_KEY
#
# Requirements (Beispiel):
# streamlit>=1.33
# openai>=1.0.0
# pandas>=2.0
# matplotlib>=3.7

import os
import re
import io
import json
import time
import base64
import sqlite3
import hashlib
import secrets
from datetime import datetime, date, timedelta

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Kassenzettel Scanner", layout="wide")

# -----------------------------
# Constants
# -----------------------------
APP_DATA_DIR = os.getenv("APP_DATA_DIR", "data")
AUTH_DB_PATH = os.path.join(APP_DATA_DIR, "auth.db")

DEFAULT_CATEGORIES = [
    "Obst & Gemüse",
    "Fleisch & Fisch",
    "Milchprodukte & Eier",
    "Brot & Backwaren",
    "Getränke",
    "Tiefkühl",
    "Konserven & Haltbares",
    "Snacks & Süßes",
    "Gewürze & Saucen",
    "Haushalt & Drogerie",
    "Sonstiges",
]

MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4o"]

# -----------------------------
# Utilities
# -----------------------------
def ensure_dirs():
    os.makedirs(APP_DATA_DIR, exist_ok=True)

def get_openai_key() -> str:
    # Secrets hat Vorrang, dann ENV
    return st.secrets.get("OPENAI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")

def safe_username(u: str) -> str:
    u = (u or "").strip().lower()
    u = re.sub(r"[^a-z0-9_\-\.]", "_", u)
    return u[:64]

def pbkdf2_hash_password(password: str, salt: bytes | None = None, iterations: int = 210_000) -> tuple[bytes, bytes]:
    if salt is None:
        salt = secrets.token_bytes(16)
    pw = (password or "").encode("utf-8")
    dk = hashlib.pbkdf2_hmac("sha256", pw, salt, iterations, dklen=32)
    return salt, dk

def pbkdf2_verify_password(password: str, salt: bytes, expected_hash: bytes, iterations: int = 210_000) -> bool:
    _, dk = pbkdf2_hash_password(password, salt=salt, iterations=iterations)
    return secrets.compare_digest(dk, expected_hash)

# -----------------------------
# Auth DB
# -----------------------------
def auth_db():
    ensure_dirs()
    conn = sqlite3.connect(AUTH_DB_PATH, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            salt BLOB NOT NULL,
            pw_hash BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

def register_user(username: str, password: str):
    username = safe_username(username)
    if len(username) < 3:
        raise ValueError("Username zu kurz (min. 3 Zeichen).")
    if len(password or "") < 8:
        raise ValueError("Passwort zu kurz (min. 8 Zeichen).")

    salt, pw_hash = pbkdf2_hash_password(password)
    conn = auth_db()
    try:
        conn.execute(
            "INSERT INTO users (username, salt, pw_hash, created_at) VALUES (?, ?, ?, ?)",
            (username, salt, pw_hash, datetime.utcnow().isoformat())
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise ValueError("Username ist bereits vergeben.")
    finally:
        conn.close()

def login_user(username: str, password: str) -> bool:
    username = safe_username(username)
    conn = auth_db()
    try:
        row = conn.execute("SELECT salt, pw_hash FROM users WHERE username = ?", (username,)).fetchone()
        if not row:
            return False
        salt, pw_hash = row[0], row[1]
        return pbkdf2_verify_password(password, salt, pw_hash)
    finally:
        conn.close()

# -----------------------------
# Per-user DB
# -----------------------------
def user_db_path(username: str) -> str:
    username = safe_username(username)
    return os.path.join(APP_DATA_DIR, f"{username}.db")

def user_db(username: str):
    ensure_dirs()
    path = user_db_path(username)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            receipt_date TEXT,
            merchant TEXT,
            address TEXT,
            currency TEXT,
            total REAL,
            source_filename TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            receipt_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            category TEXT,
            qty REAL,
            unit TEXT,
            unit_price REAL,
            total_price REAL,
            raw_line TEXT,
            FOREIGN KEY(receipt_id) REFERENCES receipts(id)
        )
    """)
    conn.commit()
    return conn

def insert_receipt_and_items(username: str, data: dict, source_filename: str | None = None) -> int:
    conn = user_db(username)
    try:
        rec_date = data.get("date")
        merchant = data.get("merchant")
        address = data.get("address")
        currency = data.get("currency") or "EUR"
        total = data.get("total")
        created_at = datetime.utcnow().isoformat()

        cur = conn.execute(
            "INSERT INTO receipts (created_at, receipt_date, merchant, address, currency, total, source_filename) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (created_at, rec_date, merchant, address, currency, total, source_filename)
        )
        receipt_id = cur.lastrowid

        items = data.get("items") or []
        for it in items:
            conn.execute("""
                INSERT INTO items (
                    receipt_id, name, category, qty, unit, unit_price, total_price, raw_line
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                receipt_id,
                (it.get("name") or "").strip(),
                it.get("category"),
                it.get("qty"),
                it.get("unit"),
                it.get("unit_price"),
                it.get("total_price"),
                it.get("raw_line"),
            ))

        conn.commit()
        return receipt_id
    finally:
        conn.close()

def load_items_df(username: str) -> pd.DataFrame:
    conn = user_db(username)
    try:
        df = pd.read_sql_query("""
            SELECT
                r.id AS receipt_id,
                r.receipt_date,
                r.merchant,
                r.currency,
                r.total AS receipt_total,
                r.created_at,
                i.name,
                i.category,
                i.qty,
                i.unit,
                i.unit_price,
                i.total_price
            FROM items i
            JOIN receipts r ON r.id = i.receipt_id
            ORDER BY r.created_at DESC, i.id ASC
        """, conn)
        return df
    finally:
        conn.close()

# -----------------------------
# Progress UI
# -----------------------------
class ProgressTracker:
    def __init__(self, title: str = "Extraktion läuft…"):
        self.pbar = st.progress(0)
        self.text = st.empty()
        self.t0 = time.time()
        self.status = None
        try:
            self.status = st.status(title, expanded=True)
        except Exception:
            self.status = None

    def update(self, pct: int, msg: str):
        pct = max(0, min(100, int(pct)))
        elapsed = time.time() - self.t0
        line = f"{msg} · {pct}% · {elapsed:0.1f}s"
        self.pbar.progress(pct)
        self.text.markdown(line)
        if self.status:
            self.status.write(line)

    def done(self, msg: str = "Fertig ✅"):
        self.update(100, msg)
        if self.status:
            self.status.update(label=msg, state="complete", expanded=False)

    def error(self, msg: str):
        if self.status:
            self.status.update(label="Fehler ❌", state="error", expanded=True)
        self.text.error(msg)

# -----------------------------
# OpenAI Vision extraction
# -----------------------------
def image_bytes_to_data_url(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def build_prompt(categories: list[str], include_categories: bool) -> tuple[str, str]:
    system = (
        "Du bist ein extrem genauer Kassenzettel-Parser. "
        "Gib die Extraktion strikt als JSON zurück: kein Markdown, kein Fließtext, keine Kommentare."
    )

    if include_categories:
        cats = ", ".join(categories)
        category_rules = (
            f'Nutze für "category" nur eine der folgenden Kategorien: [{cats}]. '
            'Wenn unklar, nimm "Sonstiges".'
        )
    else:
        category_rules = 'Setze "category" auf null.'

    user = (
        "Extrahiere Händlerdaten und alle Positionen.\n"
        "Halte Duplikate als eigene Positionen.\n"
        "Wenn Felder fehlen: null.\n"
        "Preise als Zahl (Dezimalpunkt).\n"
        f"{category_rules}\n\n"
        "JSON Schema:\n"
        "{\n"
        '  "merchant": string|null,\n'
        '  "address": string|null,\n'
        '  "currency": string|null,\n'
        '  "date": string|null,\n'
        '  "items": [\n'
        "    {\n"
        '      "name": string,\n'
        '      "category": string|null,\n'
        '      "qty": number|null,\n'
        '      "unit": string|null,\n'
        '      "unit_price": number|null,\n'
        '      "total_price": number|null,\n'
        '      "raw_line": string|null\n'
        "    }\n"
        "  ],\n"
        '  "total": number|null\n'
        "}\n"
    )
    return system, user

def extract_receipt_with_openai(image_bytes: bytes, mime: str, model: str, include_categories: bool) -> str:
    api_key = get_openai_key()
    if not api_key:
        raise RuntimeError("Kein OPENAI_API_KEY gefunden (Streamlit Secrets oder ENV).")

    client = OpenAI(api_key=api_key)
    data_url = image_bytes_to_data_url(image_bytes, mime)

    system, user = build_prompt(DEFAULT_CATEGORIES, include_categories)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": user},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ],
        temperature=0.2,
    )

    raw_text = response.choices[0].message.content
    if not raw_text:
        raise ValueError("Keine Antwort von OpenAI erhalten.")
    return raw_text

def parse_json_strict(raw_text: str) -> dict:
    raw_text = (raw_text or "").strip()

    # Isoliere JSON (falls Modell doch drumherum schreibt)
    first = raw_text.find("{")
    last = raw_text.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise ValueError("Kein JSON im Modelloutput gefunden.")

    candidate = raw_text[first:last + 1]
    data = json.loads(candidate)

    if "items" not in data or not isinstance(data["items"], list):
        raise ValueError("Ungültiges JSON: 'items' fehlt oder ist kein Array.")

    # Minimal sanity
    for it in data["items"]:
        if not isinstance(it, dict) or not (it.get("name") or "").strip():
            raise ValueError("Ungültiges JSON: Mindestens ein Item ohne gültigen 'name'.")
        # Normalize category
        cat = it.get("category")
        if cat is not None and cat not in DEFAULT_CATEGORIES:
            it["category"] = "Sonstiges"

    # Normalize currency
    if not data.get("currency"):
        data["currency"] = "EUR"

    return data

# -----------------------------
# Analytics helpers
# -----------------------------
def coerce_date(s: str | None) -> date | None:
    if not s:
        return None
    s = s.strip()
    # akzeptiere YYYY-MM-DD
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

def add_best_effort_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "receipt_date" in df.columns:
        df["receipt_date_parsed"] = df["receipt_date"].apply(coerce_date)
    else:
        df["receipt_date_parsed"] = None

    df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    df["created_date"] = df["created_at_dt"].dt.date

    df["day"] = df["receipt_date_parsed"].fillna(df["created_date"])
    return df

def item_amount(row) -> float:
    tp = row.get("total_price")
    if pd.notna(tp):
        try:
            return float(tp)
        except Exception:
            pass
    q = row.get("qty")
    up = row.get("unit_price")
    try:
        if pd.notna(q) and pd.notna(up):
            return float(q) * float(up)
    except Exception:
        pass
    return 0.0

def build_rollups(df: pd.DataFrame):
    if df.empty:
        return df, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = add_best_effort_dates(df)
    df["amount"] = df.apply(item_amount, axis=1)
    df["category"] = df["category"].fillna("Sonstiges")
    
    # Filter out rows without valid date
    df = df[df["day"].notna()].copy()
    
    daily = df.groupby(["day"], as_index=False)["amount"].sum().sort_values("day")
    cat = df.groupby(["category"], as_index=False)["amount"].sum().sort_values("amount", ascending=False)

    # weekly: ISO week - handle NaT values
    df["iso_year"] = df["day"].apply(lambda d: d.isocalendar().year if pd.notna(d) else None)
    df["iso_week"] = df["day"].apply(lambda d: d.isocalendar().week if pd.notna(d) else None)
    weekly = df.groupby(["iso_year", "iso_week"], as_index=False)["amount"].sum().sort_values(["iso_year", "iso_week"])

    # monthly
    df["month"] = df["day"].apply(lambda d: f"{d.year:04d}-{d.month:02d}" if pd.notna(d) else None)
    monthly = df.groupby(["month"], as_index=False)["amount"].sum().sort_values("month")

    return df, daily, weekly, monthly, cat

# -----------------------------
# Session state
# -----------------------------
if "auth" not in st.session_state:
    st.session_state.auth = {"logged_in": False, "username": None}
if "busy" not in st.session_state:
    st.session_state.busy = False

# -----------------------------
# Sidebar: Auth
# -----------------------------
st.sidebar.title("🔐 Zugang")

if not st.session_state.auth["logged_in"]:
    tab_login, tab_register = st.sidebar.tabs(["Login", "Registrieren"])

    with tab_login:
        u = st.text_input("Username", key="login_u")
        p = st.text_input("Passwort", type="password", key="login_p")
        if st.button("Einloggen"):
            ok = login_user(u, p)
            if ok:
                st.session_state.auth = {"logged_in": True, "username": safe_username(u)}
                st.success("Eingeloggt.")
                st.rerun()
            else:
                st.error("Login fehlgeschlagen (Username oder Passwort falsch).")

    with tab_register:
        u2 = st.text_input("Neuer Username", key="reg_u")
        p2 = st.text_input("Neues Passwort (min. 8 Zeichen)", type="password", key="reg_p")
        p3 = st.text_input("Passwort wiederholen", type="password", key="reg_p2")
        if st.button("Account erstellen"):
            if p2 != p3:
                st.error("Passwörter stimmen nicht überein.")
            else:
                try:
                    register_user(u2, p2)
                    st.success("Account erstellt. Du kannst dich jetzt einloggen.")
                except Exception as e:
                    st.error(str(e))

    st.title("🧾 Kassenzettel Scanner")
    st.info("Bitte einloggen oder registrieren, um fortzufahren.")
    st.stop()

# Logged in
username = st.session_state.auth["username"]
st.sidebar.success(f"Eingeloggt als: {username}")

if st.sidebar.button("Logout"):
    st.session_state.auth = {"logged_in": False, "username": None}
    st.rerun()

# -----------------------------
# Main navigation
# -----------------------------
st.title("🧾 Kassenzettel Scanner")
page = st.sidebar.radio("Navigation", ["Scan & Extrahieren", "Daten & Export", "Rückblicke (Charts)"], index=0)

# -----------------------------
# Page: Scan & Extract
# -----------------------------
if page == "Scan & Extrahieren":
    st.subheader("📷 Upload")
    c1, c2, c3 = st.columns([1.2, 1, 1])

    with c1:
        uploaded = st.file_uploader("Kassenzettel Foto hochladen", type=["jpg", "jpeg", "png"])
        include_categories = st.toggle("Kategorien automatisch setzen", value=True)
    with c2:
        model = st.selectbox("OpenAI Modell", MODEL_OPTIONS, index=0)
        st.caption("Tipp: gpt-4o ist genauer, gpt-4o-mini ist günstiger.")
    with c3:
        st.write("")
        st.write("")
        do_extract = st.button("Extrahieren & Speichern", disabled=st.session_state.busy)

    if uploaded:
        st.image(uploaded, caption="Vorschau", use_container_width=True)

    if uploaded and do_extract:
        st.session_state.busy = True
        prog = ProgressTracker("Kassenzettel-Extraktion")

        try:
            prog.update(5, "Upload validieren")
            image_bytes = uploaded.getvalue()
            mime = uploaded.type or "image/jpeg"
            fname = getattr(uploaded, "name", None)

            prog.update(20, "OpenAI Vision vorbereiten")

            prog.update(55, "OpenAI Vision: Extraktion läuft")
            raw = extract_receipt_with_openai(
                image_bytes=image_bytes,
                mime=mime,
                model=model,
                include_categories=include_categories,
            )

            prog.update(75, "JSON Parsing & Validierung")
            data = parse_json_strict(raw)

            prog.update(88, "In Datenbank speichern")
            receipt_id = insert_receipt_and_items(username, data, source_filename=fname)

            prog.update(95, "Ergebnis anzeigen")
            st.success(f"Gespeichert. Receipt-ID: {receipt_id}")
            st.subheader("Extrahierte Daten (JSON)")
            st.json(data)

            if data.get("items"):
                st.subheader("Positionen")
                st.dataframe(pd.DataFrame(data["items"]), use_container_width=True)

            prog.done("Extraktion abgeschlossen ✅")

        except Exception as e:
            prog.error(str(e))
            st.exception(e)

        finally:
            st.session_state.busy = False

    st.divider()
    st.subheader("ℹ️ Hinweis zur Genauigkeit")
    st.write(
        "Der Fortschrittsbalken ist step-basiert (Upload → Vision → Parsing → DB). "
        "Während des OpenAI-Calls gibt es technisch keine echten Zwischenstände – dafür ist der Step-Status stabil und ehrlich."
    )

# -----------------------------
# Page: Data & Export
# -----------------------------
elif page == "Daten & Export":
    st.subheader("🗃️ Gespeicherte Daten")
    df = load_items_df(username)

    if df.empty:
        st.info("Noch keine Daten gespeichert.")
    else:
        df2 = add_best_effort_dates(df)
        df2["amount"] = df2.apply(item_amount, axis=1)

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.metric("Positionen", f"{len(df2):,}".replace(",", "."))
        with c2:
            st.metric("Belege", f"{df2['receipt_id'].nunique():,}".replace(",", "."))
        with c3:
            st.metric("Summe (geschätzt)", f"{df2['amount'].sum():.2f} €")

        st.dataframe(df2, use_container_width=True)

        st.divider()
        st.subheader("⬇️ Export")
        export_type = st.radio("Format", ["CSV", "JSON"], horizontal=True)

        if export_type == "CSV":
            csv_bytes = df2.to_csv(index=False).encode("utf-8")
            st.download_button("CSV herunterladen", data=csv_bytes, file_name=f"{username}_items.csv", mime="text/csv")
        else:
            json_bytes = df2.to_json(orient="records", force_ascii=False).encode("utf-8")
            st.download_button("JSON herunterladen", data=json_bytes, file_name=f"{username}_items.json", mime="application/json")

# -----------------------------
# Page: Rollups & Charts
# -----------------------------
else:
    st.subheader("📊 Rückblicke")
    df = load_items_df(username)

    if df.empty:
        st.info("Noch keine Daten gespeichert.")
    else:
        df_norm, daily, weekly, monthly, cat = build_rollups(df)

        # Filter
        st.sidebar.subheader("Filter")
        days_back = st.sidebar.selectbox("Zeitraum", [7, 14, 30, 90, 365], index=2)
        cutoff = date.today() - timedelta(days=int(days_back))

        df_f = df_norm[df_norm["day"] >= cutoff].copy()
        if df_f.empty:
            st.warning("Keine Daten im gewählten Zeitraum.")
            st.stop()

        _, daily_f, weekly_f, monthly_f, cat_f = build_rollups(df_f)

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.metric("Zeitraum", f"letzte {days_back} Tage")
        with c2:
            st.metric("Summe", f"{df_f['amount'].sum():.2f} €")
        with c3:
            st.metric("Ø pro Tag", f"{(df_f['amount'].sum()/max(1, df_f['day'].nunique())):.2f} €")

        st.divider()
        st.subheader("Tagesausgaben")
        fig = plt.figure(figsize=(10, 4))
        plt.plot(pd.to_datetime(daily_f["day"]), daily_f["amount"])
        plt.xlabel("Tag")
        plt.ylabel("€")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)

        st.subheader("Kategorieverteilung")
        st.dataframe(cat_f, use_container_width=True)

        fig2 = plt.figure(figsize=(10, 4))
        plt.bar(cat_f["category"], cat_f["amount"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("€")
        plt.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig2)

        st.subheader("Monatsübersicht")
        st.dataframe(monthly_f, use_container_width=True)

        fig3 = plt.figure(figsize=(10, 4))
        plt.plot(monthly_f["month"], monthly_f["amount"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("€")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig3)

        st.divider()
        st.subheader("Detaildaten (gefiltert)")
        st.dataframe(df_f, use_container_width=True)

# -----------------------------
# Footer / Debug hints
# -----------------------------
with st.sidebar.expander("⚙️ Debug / Setup", expanded=False):
    st.write("OPENAI_API_KEY vorhanden:", "✅" if bool(get_openai_key()) else "❌")
    st.write("Auth-DB:", AUTH_DB_PATH)
    st.write("User-DB:", user_db_path(username))