# app.py - BizInsight AI (Hybrid Agent + Non-blocking LLM reasoning)
import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import threading
import time
from typing import Dict, Any, List

# ---------- Config ----------
FALLBACK_PATH = "/mnt/data/GlobalSuperstore1.xlsx"  # your uploaded file path (used if uploader not used)
LLM_MODEL = "t5-small"  # you can switch to "google/flan-t5-small" later if desired

st.set_page_config(page_title="BizInsight AI — Hybrid Agent", layout="wide")
st.title("BizInsight AI — Hybrid Agent + GenAI")
st.write("Upload CSV/XLSX or let the fallback file load. Ask anything — agent will decompose & execute subtasks.")

# ---------- File upload + fallback ----------
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
df = None
detected = {}
if uploaded_file is not None:
    try:
        if uploaded_file.name.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
        st.success(f"Loaded `{uploaded_file.name}` — {df.shape[0]} rows, {df.shape[1]} cols")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
        df = None
else:
    try:
        df = pd.read_excel(FALLBACK_PATH)
        st.info(f"Loaded fallback file `{FALLBACK_PATH}` ({df.shape[0]} rows, {df.shape[1]} cols)")
    except Exception:
        df = None

def detect_columns(df: pd.DataFrame) -> Dict[str,str]:
    """Try to find sales/product/date/price columns heuristically."""
    detected = {}
    for c in df.columns:
        lc = c.lower()
        if ("date" in lc or "order date" in lc or "ship date" in lc) and 'date' not in detected:
            detected['date'] = c
        if any(k in lc for k in ['sale','sales','amount','revenue','profit','total']) and 'sales' not in detected:
            detected['sales'] = c
        if any(k in lc for k in ['price','mrp','cost','unit price']) and 'price' not in detected:
            detected['price'] = c
        if any(k in lc for k in ['product','item','sku','product id','prod id','product name','productname','name']) and 'product' not in detected:
            detected['product'] = c
    return detected

if df is not None:
    detected = detect_columns(df)

# ---------- Helpers ----------
def safe_to_datetime(series):
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return pd.to_datetime(series.astype(str), errors="coerce")

def detect_columns(df: pd.DataFrame) -> Dict[str,str]:
    """Try to find sales/product/date/price columns heuristically."""
    detected = {}
    for c in df.columns:
        lc = c.lower()
        if ("date" in lc or "order date" in lc or "ship date" in lc) and 'date' not in detected:
            detected['date'] = c
        if any(k in lc for k in ['sale','sales','amount','revenue','profit','total']) and 'sales' not in detected:
            detected['sales'] = c
        if any(k in lc for k in ['price','mrp','cost','unit price']) and 'price' not in detected:
            detected['price'] = c
        if any(k in lc for k in ['product','item','sku','product id','prod id','product name','productname','name']) and 'product' not in detected:
            detected['product'] = c
    return detected

# ---------- Rule-based single-intent parser (fast fallback) ----------
def rule_parse_single(q: str) -> Dict[str,Any]:
    """Return {'intent':..., 'entities':{}} for a single clause."""
    text = q.lower()
    ent={}
    # top n
    m = re.search(r'top\s+(\d+)', text)
    if m: ent['top_n'] = int(m.group(1))
    # product name after 'product' or 'for'
    pm = re.search(r'product[:\s]*([A-Za-z0-9\-_ ]+)', q, flags=re.IGNORECASE)
    if pm:
        ent['product'] = pm.group(1).strip()
    # compare pattern A vs B
    vs = re.findall(r'([A-Za-z0-9\-_ ]+)\s+vs\.?\s+([A-Za-z0-9\-_ ]+)', q, flags=re.IGNORECASE)
    if vs:
        ent['compare'] = [vs[0][0].strip(), vs[0][1].strip()]

    # map phrases to intents
    if any(k in text for k in ["best product","top product","top selling","highest sales","top seller"]):
        return {"intent":"best_product","entities":ent}
    if any(k in text for k in ["worst product","low selling","lowest sales","underperform"]):
        return {"intent":"worst_product","entities":ent}
    if any(k in text for k in ["middle product","mid product","product in the middle","median product"]):
        return {"intent":"mid_product","entities":ent}
    if any(k in text for k in ["cheapest","lowest price","least price"]):
        return {"intent":"lowest_price_product","entities":ent}
    if any(k in text for k in ["which day","best day","highest day","day had"]):
        return {"intent":"max_day_sales","entities":ent}
    if any(k in text for k in ["trend","monthly trend","sales trend","show trend"]):
        return {"intent":"monthly_trend","entities":ent}
    if any(k in text for k in ["drop","decline","decrease","why did","why are"]):
        return {"intent":"detect_drops","entities":ent}
    if any(k in text for k in ["compare"," vs ","versus"]):
        return {"intent":"compare_products","entities":ent}
    if any(k in text for k in ["suggest","improve","recommend","how to increase","profit"]):
        return {"intent":"suggest_improvements","entities":ent}
    if any(k in text for k in ["summary","summarize","overview","report"]):
        return {"intent":"business_summary","entities":ent}
    # fallback
    return {"intent":"unknown","entities":ent}

# ---------- Clause splitter (naive) ----------
def split_into_clauses(query: str) -> List[str]:
    """
    Split query into clauses using ' and ', ';', ' then ', ',', ' also '.
    Keep reasonably large chunks.
    """
    # First split on ';' and ' and then '
    parts = re.split(r';|\band then\b|\bthen\b|\band also\b|\balso\b|\band\b|,', query, flags=re.IGNORECASE)
    # clean
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [query.strip()]



# ---------- Data Visualizations ----------
if df is not None and detected:
    st.header("📈 Automatic Data Visualization")

    # 1. Time-series Sales Trend
    if 'date' in detected and 'sales' in detected:
        st.subheader("Sales Trend Over Time")
        temp = df.copy()
        temp[detected['date']] = pd.to_datetime(temp[detected['date']], errors="coerce")
        trend = temp.groupby(temp[detected['date']].dt.to_period('M'))[detected['sales']].sum()
        st.line_chart(trend)

    # 2. Top 10 Products by Sales
    if 'product' in detected and 'sales' in detected:
        st.subheader("Top   Sales")
        prod_sum = df.groupby(detected['product'])[detected['sales']].sum().sort_values(ascending=False).head(10)
        st.bar_chart(prod_sum)

    # 3. Sales by Category / Segment
    # for col in ['Category', 'Segment', 'Region', 'Sub-Category']:
    #     if col in df.columns and detected.get('sales'):
    #         st.subheader(f"Sales by {col}")
    #         cat_sum = df.groupby(col)[detected['sales']].sum()
    #         st.bar_chart(cat_sum)

    # 4. Profit vs Sales Scatter Plot
    if 'profit' in [c.lower() for c in df.columns]:
        st.subheader("Profit vs Sales (Scatter)")
        import altair as alt
        st.altair_chart(
            alt.Chart(df).mark_circle(size=60).encode(
                x=detected['sales'],
                y=[c for c in df.columns if c.lower() == 'profit'][0],
                tooltip=[detected['product'], detected['sales']]
            ).interactive(),
            use_container_width=True
        )

    # 5. Region vs Segment Heatmap
    if 'Region' in df.columns and 'Segment' in df.columns and detected.get('sales'):
        st.subheader("Sales Heatmap: Region vs Segment")
        heat = df.pivot_table(values=detected['sales'], index='Region', columns='Segment', aggfunc='sum')
        st.dataframe(heat.style.background_gradient(cmap="Blues"))

    # 6. Quantity vs Sales correlation
    if 'quantity' in detected and detected.get('sales'):
        st.subheader("Quantity vs Sales Scatter")
        import altair as alt
        st.altair_chart(
            alt.Chart(df).mark_circle(size=60).encode(
                x=detected['quantity'],
                y=detected['sales'],
                tooltip=[detected['product'], detected['sales'], detected['quantity']]
            ).interactive(),
            use_container_width=True
        )


# ---------- LLM background loader (non-blocking) ----------
if "llm" not in st.session_state:
    st.session_state.llm = None
if "llm_loading" not in st.session_state:
    st.session_state.llm_loading = False
if "llm_available" not in st.session_state:
    st.session_state.llm_available = False
if "llm_error" not in st.session_state:
    st.session_state.llm_error = None

def _load_llm_background(model_name=LLM_MODEL):
    try:
        from transformers import pipeline
        st.session_state.llm_loading = True
        pipe = pipeline("text2text-generation", model=model_name)
        st.session_state.llm = pipe
        st.session_state.llm_available = True
        st.session_state.llm_error = None
    except Exception as e:
        st.session_state.llm = None
        st.session_state.llm_available = False
        st.session_state.llm_error = str(e)
    finally:
        st.session_state.llm_loading = False

def start_llm_loading(model_name=LLM_MODEL):
    if not st.session_state.llm_loading and not st.session_state.llm_available:
        t = threading.Thread(target=_load_llm_background, args=(model_name,), daemon=True)
        t.start()

# UI for LLM controls
st.sidebar.header("GenAI (LLM) controls")
if st.session_state.llm_available:
    st.sidebar.success(f"LLM loaded: {LLM_MODEL}")
    if st.sidebar.button("Unload LLM"):
        st.session_state.llm = None
        st.session_state.llm_available = False
        st.sidebar.info("LLM unloaded.")
else:
    if st.session_state.llm_loading:
        st.sidebar.info("LLM loading in background...")
    else:
        if st.sidebar.button("Load LLM (background)"):
            start_llm_loading(LLM_MODEL)
            st.sidebar.info("Loading LLM in background...")

if st.session_state.llm_error:
    st.sidebar.error("LLM error: " + str(st.session_state.llm_error))

# ---------- LLM-based decomposition (safe wrapper) ----------
def llm_decompose(query: str) -> Dict[str,Any]:
    """
    Ask LLM to decompose the query into subtasks and map them to known intents.
    Expected JSON:
    { "subtasks": [ {"intent":"best_product", "entities": {"top_n":3}}, ... ] }
    """
    if not st.session_state.llm_available or st.session_state.llm is None:
        return {"subtasks": []}
    prompt = f"""
You are an intent decomposer for a business data assistant. User asked:
\"\"\"{query}\"\"\"

Split the user's request into an ordered list of subtasks. For each subtask, choose one of these intents:
best_product, worst_product, mid_product, max_day_sales, lowest_price_product,
compare_products, monthly_trend, detect_drops, suggest_improvements, business_summary, unknown

Return ONLY JSON. Example:
{{ "subtasks": [{{"intent":"lowest_price_product","entities":{{}}}}, {{"intent":"compare_products","entities":{{"products":["A","B"]}}}} ] }}
"""
    try:
        out = st.session_state.llm(prompt, max_length=256)[0]['generated_text']
        text = out.strip()
        # pick JSON substring
        if '{' in text and '}' in text:
            j = text[text.find('{'): text.rfind('}')+1]
            parsed = json.loads(j)
            # safety check
            if isinstance(parsed, dict) and "subtasks" in parsed:
                return parsed
    except Exception:
        pass
    return {"subtasks": []}

# ---------- Tools (same as before) ----------
def tool_best_product(df, sales_col, product_col, top_n=1):
    s = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False)
    return {"type":"table","text":f"Top {top_n} product(s) by {sales_col}","data":s.head(top_n)}

def tool_worst_product(df, sales_col, product_col, top_n=1):
    s = df.groupby(product_col)[sales_col].sum().sort_values()
    return {"type":"table","text":f"Worst {top_n} product(s) by {sales_col}","data":s.head(top_n)}

def tool_mid_product(df, sales_col, product_col):
    s = df.groupby(product_col)[sales_col].sum().sort_values()
    if len(s)==0:
        return {"type":"text","text":"No product data","data":["No products found"]}
    mid_idx = len(s)//2
    return {"type":"table","text":"Middle-ranked product","data":s.iloc[mid_idx:mid_idx+1]}

def tool_monthly_trend(df, date_col, sales_col):
    tmp = df.copy()
    tmp[date_col] = safe_to_datetime(tmp[date_col])
    tmp = tmp.dropna(subset=[date_col])
    tmp['Month'] = tmp[date_col].dt.to_period('M')
    ser = tmp.groupby('Month')[sales_col].sum().sort_index()
    return {"type":"series","text":"Monthly sales","data":ser}

def tool_detect_drops(df, date_col, sales_col, threshold_pct=0.10):
    res = tool_monthly_trend(df, date_col, sales_col)
    ser = res['data']
    pct = ser.pct_change()
    drops = pct[pct <= -threshold_pct].dropna()
    if drops.empty:
        return {"type":"text","text":"No large drops detected","data":[f"No months with >= {int(threshold_pct*100)}% drop."]}
    return {"type":"series","text":"Detected drops (pct change)","data":drops}

def tool_best_day_sales(df, date_col, sales_col):
    tmp = df.copy()
    tmp[date_col] = safe_to_datetime(tmp[date_col])
    tmp = tmp.dropna(subset=[date_col])
    day_sales = tmp.groupby(tmp[date_col].dt.date)[sales_col].sum().sort_values(ascending=False)
    return {"type":"table","text":"Day with highest sales","data":day_sales.head(1)}

def tool_lowest_price_product(df, price_col, product_col):
    s = df.groupby(product_col)[price_col].mean().sort_values()
    return {"type":"table","text":"Lowest average price product","data":s.head(1)}

def tool_compare_products(df, sales_col, product_col, products: List[str]):
    tmp = df.copy()
    tmp[product_col] = tmp[product_col].astype(str)
    filtered = tmp[tmp[product_col].str.lower().isin([p.lower() for p in products])]
    if filtered.empty:
        return {"type":"text","text":"Compare products","data":[f"No rows for products {products}"]}
    s = filtered.groupby(product_col)[sales_col].sum().sort_values(ascending=False)
    return {"type":"table","text":f"Compare products {products}","data":s}

def tool_business_summary(df, sales_col=None):
    out=[]
    if sales_col and sales_col in df.columns:
        total = df[sales_col].sum()
        mean = df[sales_col].mean()
        out.append(f"Total {sales_col}: {total:.2f}")
        out.append(f"Average {sales_col}: {mean:.2f}")
    else:
        out.append("Sales column not detected.")
    return {"type":"text","text":"Business summary","data":out}

def tool_suggest_improvements(df, sales_col, product_col):
    msgs=[]
    if sales_col and sales_col in df.columns:
        mean_sales = df[sales_col].mean()
        msgs.append(f"Average sale: {mean_sales:.2f}")
        if mean_sales < 50:
            msgs.append("Consider price optimisation or upsell.")
    if product_col:
        prod_sales = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False)
        if len(prod_sales)>0:
            msgs.append(f"Top product: {prod_sales.index[0]}")
    return {"type":"text","text":"Improvement suggestions","data":msgs}

# ---------- Agent (tools registry, planner, executor) ----------
class SimpleAgent:
    def __init__(self):
        self.tools = {}
        self.plans = {}

    def register_tool(self, name, fn):
        self.tools[name] = fn

    def register_plan(self, intent, steps):
        self.plans[intent] = steps

    def decompose(self, query: str) -> List[Dict[str,Any]]:
        """Return list of subtasks: {'intent':..., 'entities': {...}}"""
        # 1) If llm available, ask it to decompose (preferred)
        if st.session_state.llm_available:
            parsed = llm_decompose(query)
            subtasks = parsed.get("subtasks", [])
            # validate and return non-empty
            if isinstance(subtasks, list) and subtasks:
                return subtasks
        # 2) fallback: split into clauses and parse each with rule_parse_single
        clauses = split_into_clauses(query)
        subtasks = []
        for c in clauses:
            p = rule_parse_single(c)
            if p.get("intent") and p["intent"]!="unknown":
                subtasks.append(p)
        # if nothing parsed, attempt single clause parse
        if not subtasks:
            p = rule_parse_single(query)
            if p.get("intent") and p["intent"]!="unknown":
                subtasks.append(p)
        return subtasks

    def plan_for(self, subtask: Dict[str,Any]):
        """Return plan (list of (tool_name, kwargs)) for a subtask dict containing intent & entities."""
        intent = subtask.get("intent")
        ent = subtask.get("entities", {})
        # map intent -> tool call(s)
        if intent == "best_product":
            return [("best_product", {"sales_col": None, "product_col": None, "top_n": ent.get("top_n",1)})]
        if intent == "worst_product":
            return [("worst_product", {"sales_col": None, "product_col": None, "top_n": ent.get("top_n",1)})]
        if intent == "mid_product":
            return [("mid_product", {"sales_col": None, "product_col": None})]
        if intent == "monthly_trend":
            return [("monthly_trend", {"date_col": None, "sales_col": None})]
        if intent == "detect_drops":
            return [("detect_drops", {"date_col": None, "sales_col": None})]
        if intent == "max_day_sales":
            return [("max_day_sales", {"date_col": None, "sales_col": None})]
        if intent == "lowest_price_product":
            return [("lowest_price_product", {"price_col": None, "product_col": None})]
        if intent == "compare_products":
            # if LLM provided product list in entities, pass it; else compare will attempt to parse later
            return [("compare_products", {"sales_col": None, "product_col": None, "products": ent.get("products") or ent.get("compare")})]
        if intent == "business_summary":
            return [("business_summary", {"sales_col": None})]
        if intent == "suggest_improvements":
            return [("suggest_improvements", {"sales_col": None, "product_col": None})]
        return []

    def execute_plan(self, plan: List[tuple], df: pd.DataFrame):
        outputs=[]
        for tool_name, kwargs in plan:
            fn = self.tools.get(tool_name)
            if not fn:
                outputs.append({"type":"text","text":f"Missing tool {tool_name}","data":[tool_name]})
                continue
            # tool kwargs are filled later by caller
            try:
                outputs.append(fn(df=df, **kwargs))
            except Exception as e:
                outputs.append({"type":"text","text":f"Tool {tool_name} error","data":[str(e)]})
        return outputs

# instantiate agent and register tools
agent = SimpleAgent()
agent.register_tool("best_product", tool_best_product)
agent.register_tool("worst_product", tool_worst_product)
agent.register_tool("mid_product", tool_mid_product)
agent.register_tool("monthly_trend", tool_monthly_trend)
agent.register_tool("detect_drops", tool_detect_drops)
agent.register_tool("max_day_sales", tool_best_day_sales)
agent.register_tool("lowest_price_product", tool_lowest_price_product)
agent.register_tool("compare_products", tool_compare_products)
agent.register_tool("business_summary", tool_business_summary)
agent.register_tool("suggest_improvements", tool_suggest_improvements)

# ---------- UI: interact ----------
if df is None:
    st.info("No data loaded. Upload a CSV/XLSX or place your file at: " + FALLBACK_PATH)
else:
    detected = detect_columns(df)
    st.sidebar.write("Detected columns:")
    st.sidebar.write(detected)

    st.header("Ask anything about your data (multi-intent supported)")
    user_q = st.text_input("Example: 'Show me cheapest product and compare its sales to the top-selling product.'")

    # allow user to start LLM loading manually
    if not st.session_state.llm_available and not st.session_state.llm_loading:
        if st.button("Load LLM (background)"):
            start_llm_loading(LLM_MODEL)
            st.info("LLM loading in background. Use rule-based parsing until it's ready.")

    if user_q:
        # 1) Decompose into subtasks (LLM preferred)
        subtasks = agent.decompose(user_q)
        if not subtasks:
            st.warning("I could not decompose the request. Try simpler phrasing. (Fallback rule parser tried.)")
        else:
            st.write("Planned subtasks:", subtasks)
            # For each subtask build a plan and execute sequentially
            final_outputs = []
            for sub in subtasks:
                plan = agent.plan_for(sub)
                # fill kwargs from detected columns and any entities
                filled_plan = []
                for tool_name, kw in plan:
                    k = dict(kw)
                    if 'sales_col' in k:
                        k['sales_col'] = detected.get('sales') or k.get('sales_col')
                    if 'product_col' in k:
                        k['product_col'] = detected.get('product') or k.get('product_col')
                    if 'date_col' in k:
                        k['date_col'] = detected.get('date') or k.get('date_col')
                    if 'price_col' in k:
                        k['price_col'] = detected.get('price') or k.get('price_col')
                    # entity extras
                    entities = sub.get('entities', {})
                    if entities.get('top_n') and 'top_n' in k:
                        k['top_n'] = entities['top_n']
                    if entities.get('products') and 'products' in k:
                        k['products'] = entities['products']
                    if entities.get('compare') and 'products' in k:
                        k['products'] = entities['compare']
                    # fallback: try to parse compare products from text if missing
                    if sub.get('intent')=='compare_products' and not k.get('products'):
                        m = re.search(r'compare\s+(.+?)\s+(and|vs|vs.)\s+(.+)', user_q, flags=re.IGNORECASE)
                        if m:
                            k['products'] = [m.group(1).strip(), m.group(3).strip()]
                    filled_plan.append((tool_name, k))
                # Execute
                out = agent.execute_plan(filled_plan, df)
                final_outputs.extend(out)

            # Render outputs
            st.markdown("### Results")
            for out in final_outputs:
                st.write("**" + out.get('text','') + "**")
                if out['type']=='table':
                    data = out['data']
                    if isinstance(data, pd.Series):
                        st.dataframe(data.reset_index().rename(columns={0:'value'}))
                        st.bar_chart(data.astype(float))
                    else:
                        st.dataframe(data)
                elif out['type']=='series':
                    ser = out['data']
                    st.line_chart(ser.astype(float))
                    st.dataframe(ser.reset_index())
                elif out['type']=='text':
                    for line in out['data']:
                        st.write("-", line)
            st.markdown("---")
            st.write("Done. You can ask another multi-step question.")

# ---------- footer tips ----------
st.markdown("---")
st.write("Tips: For best results: (1) upload a cleaned sales CSV/Excel with product, sales, date, price columns; (2) try queries like 'Show cheapest product and compare with top product', 'Show monthly trend for best product', 'Why did sales drop in March?'.")
