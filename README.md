# 🏠 Rental Assistant Chatbot

An intelligent rental assistant built with **Streamlit**, **OpenAI GPT**, **Pinecone**, and **LangChain**.  
This app allows users to perform **semantic search** on real estate listings using natural language filters.

Supports multiple filters:
- District / Area  
- Pets allowed  
- Furnished or not  
- Renovation style (Designer, Modern, Basic)  
- Price range  
- Custom wishes (e.g., balcony, elevator, etc.)

✨ Designed with a clean UI, advanced LLM explanation for results, and CSV export functionality.

---

## 🚀 Features

- 🔎 **Semantic search** using OpenAI + Pinecone vector database  
- 🧠 LLM-based explanations: “Why this result?”  
- 🧾 Export filtered results to CSV  
- ♻️ Restart search with one click  
- 🌍 Multilingual-friendly setup (future-ready)

---

## 🛠️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Margii4/rental_demo_chatbot.git
cd rental_demo_chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv311
```

#### On macOS/Linux:

```bash
source venv311/bin/activate
```

#### On Windows:

```bash
venv311\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file in the root directory

```env
OPENAI_API_KEY=your-openai-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_HOST=https://your-pinecone-project.svc.region.pinecone.io
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📦 Project Structure

```
├── app.py                  # Main Streamlit app  
├── llm_utils.py            # LLM parsing + explanation logic  
├── pinecone_utils.py       # Pinecone DB utils  
├── upsert_all.py           # Bulk upsert script  
├── parsers/  
│   ├── idealista.py        # Idealista parser  
│   └── immobiliare.py      # Immobiliare parser  
├── my_listings.json        # Sample listing data  
├── requirements.txt        # Dependencies  
└── .env                    # Your API keys (excluded from repo)
```

---

## 🌐 Deploy to Streamlit Cloud

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/rental_demo_chatbot.git
git push -u origin main
```

### 2. Deploy on Streamlit Cloud

1. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
2. Click **"New App"**
3. Select the GitHub repo
4. Set `app.py` as the entry point
5. In **Settings → Secrets**, add:

```
OPENAI_API_KEY=your-openai-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_HOST=https://your-pinecone-project.svc.region.pinecone.io
```

6. Click **Deploy**

---

## 👤 Author

**Margarita Viviers**  
GitHub: [@Margii4](https://github.com/Margii4)
