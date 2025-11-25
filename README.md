# BizInsight AI — Autonomous Business Data Analyzer

BizInsight AI is a free, beginner-friendly **GenAI + Agentic AI project** that 
automatically analyzes business data and answers natural language questions like:

- "Which product is underperforming?"
- "Why did sales drop in March?"
- "Which product is in the middle?"
- "Show monthly sales trend"
- "Which day had the highest sales?"
- "Which product is cheapest?"

It uses:

✔ Python  
✔ Streamlit  
✔ Agentic Planning  
✔ Local LLM Intent Parsing  
✔ Rule-based fallback  
✔ Automatic data visualizations  

---

## 🚀 Features

### 1. **Data Upload**
Upload any CSV or Excel file (e.g., sales or transactions).

### 2. **Automatic Business Insights**
The system automatically detects:

- Sales column  
- Product column  
- Date column  
- Price column  
- Quantity  
- Region, Segment, Category  

Then generates:

- Sales trend
- Top 10 products
- Monthly trend
- Profit vs sales scatter
- Region × Segment heatmap
- Quantity vs sales chart

### 3. **Agentic Query System**
Ask questions in plain English:

"Show me cheapest product"
"Compare A vs B"
"Which day had max sales?"
"Which product is in the middle?"
"Suggest improvements"


The agent will:

- Parse intent  
- Plan tools  
- Execute analysis  
- Generate charts  
- Respond in simple English  

### 4. **Hybrid Engine**
Uses both:

- Local LLM (t5-small) for natural language
- Agent Tools for real data analysis

Fast + Smart.

---

## 🧪 Sample Data for Testing

A sample Excel file is included:

sample_data/GlobalSuperstore1.xlsx


Users can test the full system without uploading their own data.

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
