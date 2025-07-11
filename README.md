# 🏠 Rental Demo Chatbot

A multilingual rental assistant chatbot built with **Python**, **Streamlit**, and **LLM-ready architecture**. It simulates intelligent apartment search with semantic matching, advanced filtering, CSV export, and “Why this result?” explanations powered by OpenAI.

---

## 🚀 Demo

Hosted on [Streamlit Cloud](#) 
### Preview

![Rental Bot UI](screenshots/search_ui.png)  
![LLM Explanation](screenshots/why_this_result.png)

---

## ✨ Features

- 🔍 **Smart Semantic Search** – using OpenAI and Pinecone
- 💬 **"Why This Result?"** – explained by GPT in natural language
- 🌍 **Multilingual Interface** – English 🇬🇧 and Italian 🇮🇹
- 📑 **Editable JSON Listings** – you control the data
- 📤 **CSV Export** – one-click download
- 🔁 **Restart Button** – clear search instantly

---

## 🛠 Tech Stack

- Python 3.11
- Streamlit
- OpenAI GPT-3.5
- Pinecone (Vector Database)
- LangChain (optional)
- JSON / Pandas
- Async I/O

---

## 📁 Project Structure

```bash
rental_demo_chatbot/
├── app.py                  # Main Streamlit UI logic
├── llm_utils.py            # LLM-based parsing and explanations
├── pinecone_utils.py       # Vector database interactions
├── parsers/
│   └── json_parser.py      # Custom JSON loader
├── my_listings.json        # Your rental listings
├── requirements.txt
└── README.md


💻 Local Setup

1. Clone the repo
git clone https://github.com/Margii4/rental_demo_chatbot.git
cd rental_demo_chatbot

2. Create and activate a virtual environment
python -m venv venv311
source venv311/bin/activate   # On Windows: venv311\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Add your .env file
OPENAI_API_KEY=your-openai-key
PINECONE_API_KEY=your-pinecone-key
PINECONE_HOST=https://rental-listings-j2d462y.svc.aped-4627-b74a.pinecone.io

5. Run the app
streamlit run app.py

---

📊 Potential Business Impact
| Metric                     | Estimate                   |
| -------------------------- | -------------------------- |
| Automated support volume   | \~40–50% of queries        |
| Avg support salary (Italy) | \~€27,000/year             |
| Est. savings (5 FTE)       | €67.5k–€78k/year           |
| Supported languages        | 🇬🇧 English, 🇮🇹 Italian |

---

🤝 Author
Margarita Viviers
GitHub: @Margii4
LinkedIn: linkedin.com/in/margarita-viviers-03b047362