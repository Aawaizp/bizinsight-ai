# 🚀 BizInsight AI — Autonomous Business Data Analyzer

BizInsight AI is a **GenAI + Agentic AI powered business intelligence tool** that helps users analyze CSV/Excel business datasets without writing any code.

Simply upload your dataset and ask questions in plain English like:

- *“Which product is underperforming?”*  
- *“Show monthly sales trend.”*  
- *“Which product is the cheapest?”*  
- *“Which day had the highest sales?”*  
- *“Show me the best and worst products.”*  

BizInsight AI automatically:
✅ Detects key columns (sales, product, price, dates)  
✅ Generates business insights  
✅ Creates smart charts  
✅ Uses an **Agentic AI planner** to call correct data tools  
✅ Uses **LLM-based natural language parsing**  
✅ Explains results in plain English  

---

## 🌐 Live Demo

👉 **Try the deployed app:**  
🔗 https://bizinsight-ai-kdbvmmr7c5mt8nfdch2z4e.streamlit.app/

---

## 📊 Sample Dataset

For testing, you may use the included dataset:

📄 **GlobalSuperstore1.xlsx**  
(Uploaded in repo)

This dataset includes:
- Product info  
- Order dates  
- Sales  
- Shipping cost  
- Customer + region details  

---

## ⚙️ Features

### 🔍 **1. Automatic Data Understanding**
- Auto-detects:
  - Sales column  
  - Product column  
  - Price column  
  - Date column  
- Shows summary stats  
- Shows best/worst products  
- Finds monthly sales trends  

### 🤖 **2. Agentic AI Engine**
A custom-built agent system:
- Interprets user intent  
- Maps intent → Tools  
- Executes step-by-step plans  
- Handles:
  - best product  
  - worst product  
  - mid product  
  - monthly trend  
  - drop detection  
  - cheapest product  
  - product comparison  
  - highest sales day  
  - improvement suggestions  

### 🧠 **3. LLM Integration (T5 Small)**
- Helps interpret complex natural language  
- Works as intent parser fallback  
- Ensures flexible user queries  

### 📈 **4. Visual Insights**
- Auto-generated:
  - Line charts  
  - Bar charts  
  - Tables  
  - Summaries  

---

## 🛠️ Tech Stack

| Component | Technology |
|----------|------------|
| Frontend UI | Streamlit |
| Data Processing | Pandas, NumPy |
| ML / LLM | Transformers (T5-small), Torch |
| Agent System | Custom Python-based planner & tool executor |
| Deployment | Streamlit Cloud |
| File Support | CSV, XLSX, XLS |

---

## 💡 Example Questions You Can Ask

Try any of these:

“Which product is in the middle?”

“Which day had highest sales?”

“Which product is cheapest?”

“Show me monthly sales trend.”

“Show all products from lowest to highest price.”

“Compare product A vs B.”

“Why did sales drop in any month?”

“Which region performs best?”
