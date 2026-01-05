import os
import io
import json
import re
import sqlite3
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
from PIL import Image
import fitz  # PyMuPDF

from openai import OpenAI

# Password hashing
import bcrypt


# -----------------------------
# App Config
# -----------------------------
APP_TITLE = "Receipt Radar – Kassenzettel Scanner & Auswertung"
AUTH_DB_PATH = "auth.db"
DATA_DIR = "data"

DEFAULT_CATEGORIES = [
    "Obst & Gemüse",
    "Fleisch & Fisch",
    "Milchprodukte",
    "Backwaren",
    "Getränke",
    "Snacks & Süßes",
    "Tiefkühl",
    "Haushalt",
    "Hygiene & Kosmetik",
    "Baby & Kind",
    "Tierbedarf",
    "Drogerie/Apotheke",
    "Sonstiges",
]

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5.2")


# -----------------------------
# Utils
# -----------------------------
def safe_api_key() -> str:
    return os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

def ensure_data_dir() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

def now_utc_iso() -> str:
    return datetime.utcnow().isoformat()

def parse_date(s: Any) -> Optional[str]:
    if not s:
        return None
    if isinstance(s, (datetime, date)):
        return (s.date() if isinstance(s, datetime) else s).isoformat()
    txt = str(s).strip()
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(txt, fmt).date().isoformat()
        except Exception:
            pass
    return None

def money_to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    s = s.replace("€", "").replace("EUR", "").strip()
    # German format "12,34"
    s = s.replace(".", "").replace(",", ".") if re.search(r"\d+,\d{2}", s) else s
    s = re.sub(r"[^\d\.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return None

def week_start(d: date) -> date:
    return d - timedelta(days=d.weekday())


# -----------------------------
# Auth DB (global)
# -----------------------------
def auth_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(AUTH_DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_auth_db() -> None:
    conn = auth_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            password_hash BLOB NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()

def bcrypt_hash_password(password: str) -> bytes:
    pw_bytes = password.encode("utf-8")
    return bcrypt.hashpw(pw_bytes, bcrypt.gensalt(rounds=12))

def bcrypt_verify_password(password: str, pw_hash: bytes) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), pw_hash)
    except Exception:
        return False

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    conn = auth_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, email, password_hash, is_admin, created_at FROM users WHERE email = ?", (email.lower().strip(),))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "email": row[1],
        "password_hash": row[2],
        "is_admin": bool(row[3]),
        "created_at": row[4],
    }

def create_user(email: str, password: str, is_admin: bool = False) -> int:
    email = email.lower().strip()
    pw_hash = bcrypt_hash_password(password)
    conn = auth_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO users (email, password_hash, is_admin, created_at) VALUES (?, ?, ?, ?)",
        (email, pw_hash, 1 if is_admin else 0, now_utc_iso()),
    )
    conn.commit()
    uid = cur.lastrowid
    conn.close()
    return int(uid)

def ensure_bootstrap_admin() -> None:
    """
    Admin wird aus Secrets/ENV gezogen und beim ersten Start angelegt.
    Dadurch ist Registration öffentlich NICHT nötig.
    """
    admin_email = os.getenv("ADMIN_EMAIL") or st.secrets.get("ADMIN_EMAIL", "")
    admin_pw = os.getenv("ADMIN_PASSWORD") or st.secrets.get("ADMIN_PASSWORD", "")
    if not admin_email or not admin_pw:
        # Kein Admin konfiguriert -> App kann trotzdem laufen,
        # aber du kannst keine User anlegen.
        return

    if get_user_by_email(admin_email) is None:
        create_user(admin_email, admin_pw, is_admin=True)


# -----------------------------
# Per-User Receipt DB
# -----------------------------
def user_db_path(user_id: int) -> str:
    ensure_data_dir()
    return os.path.join(DATA_DIR, f"user_{user_id}_receipts.db")

def user_db_conn(user_id: int) -> sqlite3.Connection:
    conn = sqlite3.connect(user_db_path(user_id), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_user_db(user_id: int) -> None:
    conn = user_db_conn(user_id)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS receipts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            merchant TEXT,
            receipt_date TEXT,
            currency TEXT,
            total REAL,
            subtotal REAL,
            tax REAL,
            confidence REAL,
            source_filename TEXT,
            raw_json TEXT
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS line_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            receipt_id INTEGER NOT NULL,
            item_name TEXT,
            quantity REAL,
            unit_price REAL,
            line_total REAL,
            category TEXT,
            FOREIGN KEY(receipt_id) REFERENCES receipts(id) ON DELETE CASCADE
        );
        """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_receipts_date ON receipts(receipt_date);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_items_category ON line_items(category);")

    conn.commit()
    conn.close()

def insert_receipt(
    user_id: int,
    merchant: Optional[str],
    receipt_date: Optional[str],
    currency: Optional[str],
    total: Optional[float],
    subtotal: Optional[float],
    tax: Optional[float],
    confidence: Optional[float],
    source_filename: Optional[str],
    raw_json: Dict[str, Any],
    items: List[Dict[str, Any]],
) -> int:
    init_user_db(user_id)
    conn = user_db_conn(user_id)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO receipts (created_at, merchant, receipt_date, currency, total, subtotal, tax, confidence, source_filename, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            now_utc_iso(),
            merchant,
            receipt_date,
            currency,
            total,
            subtotal,
            tax,
            confidence,
            source_filename,
            json.dumps(raw_json, ensure_ascii=False),
        ),
    )
    receipt_id = cur.lastrowid

    for it in items:
        cur.execute(
            """
            INSERT INTO line_items (receipt_id, item_name, quantity, unit_price, line_total, category)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                receipt_id,
                it.get("item_name"),
                it.get("quantity"),
                it.get("unit_price"),
                it.get("line_total"),
                it.get("category"),
            ),
        )

    conn.commit()
    conn.close()
    return int(receipt_id)

def load_receipts_df(user_id: int) -> pd.DataFrame:
    init_user_db(user_id)
    conn = user_db_conn(user_id)
    df = pd.read_sql_query(
        """
        SELECT id, created_at, merchant, receipt_date, currency, total, subtotal, tax, confidence, source_filename
        FROM receipts
        ORDER BY COALESCE(receipt_date, created_at) DESC
        """,
        conn,
    )
    conn.close()
    return df

def load_items_df(user_id: int, receipt_id: Optional[int] = None) -> pd.DataFrame:
    init_user_db(user_id)
    conn = user_db_conn(user_id)
    if receipt_id is None:
        df = pd.read_sql_query(
            """
            SELECT li.id, li.receipt_id, r.receipt_date, r.merchant, li.item_name, li.quantity, li.unit_price, li.line_total, li.category
            FROM line_items li
            JOIN receipts r ON r.id = li.receipt_id
            ORDER BY COALESCE(r.receipt_date, r.created_at) DESC, li.id ASC
            """,
            conn,
        )
    else:
        df = pd.read_sql_query(
            """
            SELECT li.id, li.receipt_id, r.receipt_date, r.merchant, li.item_name, li.quantity, li.unit_price, li.line_total, li.category
            FROM line_items li
            JOIN receipts r ON r.id = li.receipt_id
            WHERE li.receipt_id = ?
            ORDER BY li.id ASC
            """,
            conn,
            params=(receipt_id,),
        )
    conn.close()
    return df

def update_item_category(user_id: int, item_id: int, category: str) -> None:
    init_user_db(user_id)
    conn = user_db_conn(user_id)
    conn.execute("UPDATE line_items SET category = ? WHERE id = ?", (category, item_id))
    conn.commit()
    conn.close()

def delete_receipt(user_id: int, receipt_id: int) -> None:
    init_user_db(user_id)
    conn = user_db_conn(user_id)
    conn.execute("DELETE FROM receipts WHERE id = ?", (receipt_id,))
    conn.commit()
    conn.close()


# -----------------------------
# File helpers
# -----------------------------
def pdf_to_pil_first_page(pdf_bytes: bytes, zoom: float = 2.0) -> Image.Image:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    doc.close()
    return img

def pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def b64(b: bytes) -> str:
    import base64
    return base64.b64encode(b).decode("utf-8")


# -----------------------------
# OpenAI extraction
# -----------------------------
def build_receipt_prompt(categories: List[str]) -> str:
    schema_hint = {
        "merchant": "string | null",
        "receipt_date": "YYYY-MM-DD | null",
        "currency": "string | null (e.g. EUR)",
        "totals": {"total": "number | null", "subtotal": "number | null", "tax": "number | null"},
        "items": [
            {"item_name": "string", "quantity": "number | null", "unit_price": "number | null",
             "line_total": "number | null", "category": f"one of {categories}"}
        ],
        "confidence": "number 0..1",
        "notes": "string | null",
    }

    return f"""
Du bist ein präziser Beleg-Parser. Extrahiere aus dem Kassenzettel folgende Felder und gib AUSSCHLIESSLICH gültiges JSON zurück (kein Markdown, keine Erklärungen).

Regeln:
- Nutze Dezimalpunkt im JSON (12.34), keine Kommas.
- Datum als YYYY-MM-DD, wenn erkennbar; sonst null.
- Währung als z.B. EUR; sonst null.
- Artikelpositionen: so gut wie möglich. Wenn Mengen/Einzelpreis fehlen, setze null.
- Kategorie: wähle die passendste aus der Liste. Wenn unsicher: "Sonstiges".
- confidence: 0..1.

Kategorien:
{categories}

Schema-Hinweis:
{json.dumps(schema_hint, ensure_ascii=False)}

WICHTIG: Ausgabe muss parsebares JSON sein.
""".strip()

def extract_json_object(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError("Konnte kein gültiges JSON aus der Modellantwort parsen.")

def extract_receipt_with_openai(image_png_bytes: bytes, categories: List[str], api_key: str) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key)
    prompt = build_receipt_prompt(categories)

    resp = client.responses.create(
        model=MODEL_NAME,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/png;base64,{b64(image_png_bytes)}"},
            ],
        }],
    )
    text = resp.output_text.strip()
    return extract_json_object(text)

def normalize_extraction(raw: Dict[str, Any], categories: List[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    merchant = raw.get("merchant")
    receipt_date = parse_date(raw.get("receipt_date"))
    currency = raw.get("currency") or "EUR"

    totals = raw.get("totals") or {}
    total = money_to_float(totals.get("total"))
    subtotal = money_to_float(totals.get("subtotal"))
    tax = money_to_float(totals.get("tax"))

    confidence = raw.get("confidence")
    try:
        confidence = float(confidence) if confidence is not None else None
    except Exception:
        confidence = None

    items_in = raw.get("items") or []
    items_out: List[Dict[str, Any]] = []

    for it in items_in:
        name = (it.get("item_name") or "").strip()
        if not name:
            continue
        qty = money_to_float(it.get("quantity"))
        unit = money_to_float(it.get("unit_price"))
        line = money_to_float(it.get("line_total"))
        cat = it.get("category") or "Sonstiges"
        if cat not in categories:
            cat = "Sonstiges"

        items_out.append({
            "item_name": name,
            "quantity": qty,
            "unit_price": unit,
            "line_total": line,
            "category": cat,
        })

    normalized = {
        "merchant": merchant,
        "receipt_date": receipt_date,
        "currency": currency,
        "total": total,
        "subtotal": subtotal,
        "tax": tax,
        "confidence": confidence,
    }
    return normalized, items_out


# -----------------------------
# Auth UI
# -----------------------------
def logout():
    st.session_state.pop("auth_user", None)
    st.session_state.pop("auth_is_admin", None)
    st.session_state.pop("auth_user_id", None)
    st.rerun()

def require_login() -> Tuple[int, bool, str]:
    """
    Returns (user_id, is_admin, email). Stops app if not logged in.
    """
    if st.session_state.get("auth_user_id"):
        return int(st.session_state["auth_user_id"]), bool(st.session_state["auth_is_admin"]), str(st.session_state["auth_user"])

    st.title(APP_TITLE)
    st.subheader("🔐 Login")

    with st.form("login_form"):
        email = st.text_input("E-Mail", placeholder="name@domain.de").strip().lower()
        pw = st.text_input("Passwort", type="password")
        submitted = st.form_submit_button("Einloggen")

    if submitted:
        user = get_user_by_email(email)
        if not user or not bcrypt_verify_password(pw, user["password_hash"]):
            st.error("Login fehlgeschlagen. E-Mail oder Passwort falsch.")
            st.stop()

        st.session_state["auth_user_id"] = int(user["id"])
        st.session_state["auth_user"] = user["email"]
        st.session_state["auth_is_admin"] = bool(user["is_admin"])
        st.rerun()

    st.info("Kein öffentlicher Zugriff. Nur vorhandene Nutzer können sich einloggen.")
    st.stop()


# -----------------------------
# Main UI
# -----------------------------
def set_page():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

def sidebar_settings() -> Tuple[List[str], str]:
    st.sidebar.header("Einstellungen")

    api_key = safe_api_key()
    if not api_key:
        api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Wird nicht gespeichert (nur Session).")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Kategorien")
    categories_text = st.sidebar.text_area(
        "Eine Kategorie pro Zeile",
        value="\n".join(DEFAULT_CATEGORIES),
        height=220,
    )
    categories = [c.strip() for c in categories_text.splitlines() if c.strip()]
    if "Sonstiges" not in categories:
        categories.append("Sonstiges")

    return categories, api_key

def nav(is_admin: bool) -> str:
    pages = ["📸 Scan & Import", "📊 Dashboard", "🧾 Belege & Positionen"]
    if is_admin:
        pages.append("👤 Admin")
    pages.append("⚙️ Daten")
    return st.sidebar.radio("Navigation", pages, index=0)

def admin_view():
    st.subheader("👤 Admin – User anlegen")
    st.caption("Best Practice: Registrierung ist aus. Admin legt Accounts an.")

    with st.form("create_user"):
        email = st.text_input("Neue User E-Mail").strip().lower()
        pw = st.text_input("Passwort", type="password")
        is_admin = st.checkbox("Admin-Rechte", value=False)
        ok = st.form_submit_button("User erstellen")

    if ok:
        if not email or not pw:
            st.error("Bitte E-Mail und Passwort setzen.")
            return
        if get_user_by_email(email) is not None:
            st.error("User existiert bereits.")
            return
        uid = create_user(email, pw, is_admin=is_admin)
        st.success(f"User angelegt. ID: {uid}")

def scan_import_view(user_id: int, categories: List[str], api_key: str):
    st.subheader("📸 Scan & Import")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        uploaded = st.file_uploader("Kassenzettel hochladen (JPG/PNG/PDF)", type=["jpg", "jpeg", "png", "pdf"])
        if uploaded:
            filename = uploaded.name
            bytes_data = uploaded.read()

            if filename.lower().endswith(".pdf"):
                try:
                    img = pdf_to_pil_first_page(bytes_data, zoom=2.2)
                except Exception as e:
                    st.error(f"PDF konnte nicht gerendert werden: {e}")
                    return
            else:
                img = Image.open(io.BytesIO(bytes_data)).convert("RGB")

            st.image(img, caption="Vorschau", use_container_width=True)

            if st.button("🔎 Mit KI auslesen", use_container_width=True, disabled=not api_key):
                with st.spinner("Extrahiere Daten…"):
                    try:
                        png_bytes = pil_to_png_bytes(img)
                        raw = extract_receipt_with_openai(png_bytes, categories, api_key)
                        normalized, items = normalize_extraction(raw, categories)

                        st.session_state["last_raw"] = raw
                        st.session_state["last_norm"] = normalized
                        st.session_state["last_items"] = items
                        st.session_state["last_filename"] = filename

                        st.success("Fertig. Bitte prüfen und speichern.")
                    except Exception as e:
                        st.error(f"Extraktion fehlgeschlagen: {e}")

    with col2:
        st.markdown("### Ergebnis (prüfen & speichern)")
        norm = st.session_state.get("last_norm")
        items = st.session_state.get("last_items")
        raw = st.session_state.get("last_raw")
        filename = st.session_state.get("last_filename")

        if not norm:
            st.info("Noch kein Ergebnis. Lade einen Beleg hoch und starte die KI-Extraktion.")
            return

        merchant = st.text_input("Händler", value=norm.get("merchant") or "")
        receipt_date = st.text_input("Datum (YYYY-MM-DD)", value=norm.get("receipt_date") or "")
        currency = st.text_input("Währung", value=norm.get("currency") or "EUR")

        total = st.number_input("Gesamt", value=float(norm["total"]) if norm.get("total") is not None else 0.0, step=0.01)
        subtotal = st.number_input("Zwischensumme", value=float(norm["subtotal"]) if norm.get("subtotal") is not None else 0.0, step=0.01)
        tax = st.number_input("MwSt/Steuer", value=float(norm["tax"]) if norm.get("tax") is not None else 0.0, step=0.01)

        df = pd.DataFrame(items or [])
        if df.empty:
            st.warning("Keine Positionen erkannt. Du kannst trotzdem speichern.")
            df = pd.DataFrame(columns=["item_name", "quantity", "unit_price", "line_total", "category"])

        edited = st.data_editor(
            df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={"category": st.column_config.SelectboxColumn("category", options=categories)},
        )

        colA, colB = st.columns([1, 1])
        with colA:
            if st.button("💾 In meine DB speichern", use_container_width=True):
                rid = insert_receipt(
                    user_id=user_id,
                    merchant=merchant.strip() or None,
                    receipt_date=parse_date(receipt_date),
                    currency=currency.strip() or None,
                    total=float(total) if total is not None else None,
                    subtotal=float(subtotal) if subtotal is not None else None,
                    tax=float(tax) if tax is not None else None,
                    confidence=money_to_float(norm.get("confidence")),
                    source_filename=filename,
                    raw_json=raw or {},
                    items=edited.to_dict(orient="records"),
                )
                st.success(f"Gespeichert (Receipt ID: {rid}).")

                # clear
                for k in ["last_raw", "last_norm", "last_items", "last_filename"]:
                    st.session_state.pop(k, None)

        with colB:
            with st.expander("Raw JSON (Debug)"):
                st.json(raw)

def dashboard_view(user_id: int):
    st.subheader("📊 Dashboard")
    receipts = load_receipts_df(user_id)
    items = load_items_df(user_id)

    if receipts.empty:
        st.info("Noch keine Daten. Importiere zuerst Kassenzettel.")
        return

    items["receipt_date"] = pd.to_datetime(items["receipt_date"], errors="coerce")
    min_d = items["receipt_date"].min()
    max_d = items["receipt_date"].max()
    if pd.isna(min_d) or pd.isna(max_d):
        min_d = pd.Timestamp(date.today() - timedelta(days=30))
        max_d = pd.Timestamp(date.today())

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        start = st.date_input("Von", value=min_d.date())
    with col2:
        end = st.date_input("Bis", value=max_d.date())
    with col3:
        group = st.selectbox("Ansicht", ["Tag", "Woche", "Monat"], index=1)

    mask = (items["receipt_date"] >= pd.Timestamp(start)) & (items["receipt_date"] <= pd.Timestamp(end))
    df = items.loc[mask].copy()

    if df.empty:
        st.info("Keine Daten im Zeitraum.")
        return

    df["line_total"] = pd.to_numeric(df["line_total"], errors="coerce").fillna(0.0)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Summe", f"{df['line_total'].sum():.2f} €")
    k2.metric("Anzahl Positionen", f"{len(df)}")
    k3.metric("Anzahl Belege", f"{df['receipt_id'].nunique()}")
    k4.metric("Ø pro Beleg", f"{(df['line_total'].sum() / max(df['receipt_id'].nunique(),1)):.2f} €")

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        tmp = df.dropna(subset=["receipt_date"]).copy()
        if tmp.empty:
            st.warning("Keine gültigen Daten für Zeitverlauf (Datum fehlt).")
        else:
            if group == "Tag":
                tmp["k"] = tmp["receipt_date"].dt.date.astype(str)
            elif group == "Woche":
                tmp["k"] = tmp["receipt_date"].dt.date.apply(week_start).astype(str)
            else:
                tmp["k"] = tmp["receipt_date"].dt.to_period("M").astype(str)

            agg = tmp.groupby("k", as_index=False)["line_total"].sum().rename(columns={"line_total": "sum"})
            st.line_chart(agg.set_index("k")["sum"])

    with right:
        cat = df.groupby("category", as_index=False)["line_total"].sum().sort_values("line_total", ascending=False)
        st.bar_chart(cat.set_index("category")["line_total"])

    st.markdown("### Export")
    export_df = df.copy()
    export_df["receipt_date"] = export_df["receipt_date"].dt.date.astype(str)
    st.download_button(
        "⬇️ CSV Export (Positionen)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="receipt_items_export.csv",
        mime="text/csv",
        use_container_width=True,
    )

def receipts_items_view(user_id: int, categories: List[str]):
    st.subheader("🧾 Belege & Positionen")
    receipts = load_receipts_df(user_id)
    if receipts.empty:
        st.info("Noch keine Belege vorhanden.")
        return

    st.dataframe(receipts, use_container_width=True)

    st.markdown("---")
    receipt_id = st.number_input("Receipt ID", min_value=1, step=1)
    items = load_items_df(user_id, int(receipt_id))
    if items.empty:
        st.info("Keine Positionen gefunden (oder Receipt ID existiert nicht).")
        return

    st.dataframe(items, use_container_width=True)

    with st.expander("Kategorie ändern"):
        item_id = st.number_input("Item ID", min_value=1, step=1)
        new_cat = st.selectbox("Neue Kategorie", options=categories, index=categories.index("Sonstiges") if "Sonstiges" in categories else 0)
        if st.button("✅ Speichern"):
            update_item_category(user_id, int(item_id), new_cat)
            st.success("Kategorie aktualisiert.")

def data_view(user_id: int):
    st.subheader("⚙️ Daten")
    st.caption("Du arbeitest in deiner eigenen Datenbankdatei:")
    st.code(user_db_path(user_id))

    receipts = load_receipts_df(user_id)
    items = load_items_df(user_id)

    c1, c2 = st.columns(2)
    c1.metric("Belege", f"{len(receipts)}")
    c2.metric("Positionen", f"{len(items)}")

    st.markdown("### Gefährlich: Beleg löschen")
    rid = st.number_input("Receipt ID zum Löschen", min_value=1, step=1, key="del_rid")
    if st.button("🗑️ Löschen", type="secondary"):
        delete_receipt(user_id, int(rid))
        st.success("Gelöscht (falls vorhanden).")


def main():
    set_page()
    init_auth_db()
    ensure_bootstrap_admin()
    ensure_data_dir()

    # Hard gate
    user_id, is_admin, email = require_login()

    st.sidebar.markdown(f"**Angemeldet:** {email}")
    if st.sidebar.button("Logout"):
        logout()

    categories, api_key = sidebar_settings()
    page = nav(is_admin)

    if page == "📸 Scan & Import":
        scan_import_view(user_id, categories, api_key)
    elif page == "📊 Dashboard":
        dashboard_view(user_id)
    elif page == "🧾 Belege & Positionen":
        receipts_items_view(user_id, categories)
    elif page == "👤 Admin":
        admin_view()
    else:
        data_view(user_id)


if __name__ == "__main__":
    main()