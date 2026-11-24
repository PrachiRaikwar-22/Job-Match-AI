
# 📄 Job Match AI

A mini project that compares your resume with a job description (JD), shows how well you
match, highlights matched and missing keywords, and uses an LLM (Groq) to give **honest
suggestions** on what to add or learn.

Built with **Python + Streamlit** only. No JavaScript and no FastAPI needed.
🚀 **Live App:** [Try Job Match AI](https://prachiraikwar-22-job-match-ai-app-1fdn7i.streamlit.app/)

## Features

- Upload your resume as a **PDF** (or TXT).
- **Paste** the job description as text.
- NLP preprocessing: lowercase, clean, stop-word removal, light stemming.
- **TF-IDF + cosine similarity** score, plus a keyword coverage score and an overall match %.
- ✅ **Matched** and ❌ **missing** keywords as chips, with matched words highlighted in the resume and JD.
- 💡 **AI suggestions** (RAG-style): the resume sections most relevant to the JD are retrieved
  with TF-IDF cosine and sent to the LLM, which returns:
  - **You can add these**: missing keywords where your resume already shows related experience, with evidence and a truthful bullet.
  - **Real gaps**: skills to learn, with project ideas. Not to be added to the resume yet.
  - **Other improvements**: structure, action verbs, impact.
- The AI never writes a new resume and is told not to invent skills or experience.
- 🌙 / ☀️ **Day / night toggle** at the top right. It shows the moon in day mode and the sun in
  night mode. The first load follows your system theme.

## Project structure

```
JobMatchAI/
├── app.py              # Streamlit UI, theme toggle, results, suggestions
├── core/
│   ├── parser.py       # PDF / TXT -> clean text
│   ├── nlp.py          # preprocessing, TF-IDF, cosine, keywords, highlighting, retrieval
│   └── llm.py          # Groq call + suggestion prompt
├── requirements.txt
├── .env.example        # copy to .env and add your key
└── README.md
```

## Setup (VS Code)

1. Install **Python 3.10+** and **VS Code** with the *Python* extension.
2. In VS Code: **File → Open Folder** and choose the `Job Match AI` folder.
3. Open a terminal: **Terminal → New Terminal**.
4. Create and activate a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac / Linux: `python3 -m venv venv` then `source venv/bin/activate`
5. Install the packages: `pip install -r requirements.txt`
6. Get a free key at https://console.groq.com/keys, then copy `.env.example` to `.env` and edit it:
   ```
   GROQ_API_KEY=gsk_your_key_here
   # optional, default is openai/gpt-oss-120b
   # GROQ_MODEL=openai/gpt-oss-120b
   ```
   The key stays in the backend. There is no key box in the UI.
7. Run the app: `streamlit run app.py`. It opens at http://localhost:8501.

> The score and keyword analysis work without any key. Only the 💡 suggestions need `GROQ_API_KEY`.
> Never share or upload your `.env` file.

## How to use

1. Upload your resume PDF.
2. Paste the job description.
3. Click **🔍 Analyze match** to see the scores and the matched and missing keywords.
4. Click **💡 Get suggestions** to see what you can honestly add and what to learn.

## How it works

1. **Parse**: `pypdf` extracts the text from the PDF.
2. **Preprocess**: the text is lowercased and cleaned, stop words are removed and words are lightly stemmed.
3. **Score**:
   - TF-IDF vectors for the resume and the JD, with **cosine similarity** between them.
   - Keywords are the known skills found in the JD plus the top TF-IDF words. Coverage is the share of those found in the resume.
   - Overall match = 40% cosine (scaled) + 60% keyword coverage.
4. **Retrieve (RAG)**: the resume is split into chunks, and the chunks most similar to the JD are picked by TF-IDF cosine.
5. **Suggest**: the retrieved chunks, the full resume, the JD and the matched and missing lists go to Groq, which returns the suggestions.

## Customize

- **More skills:** add terms to the `SKILLS` set at the top of `core/nlp.py`.
- **Number of keywords:** change `TOP_N` in `app.py`.
- **Score weights:** edit the `final = 0.4 * cos_scaled + 0.6 * kw` line in `core/nlp.py`.
- **Another model:** set `GROQ_MODEL` in `.env`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `GROQ_API_KEY is missing` | Make sure the file is named exactly `.env` (not `.env.txt`), sits next to `app.py`, then restart Streamlit. |
| `module 'core.llm' has no attribute ...` | `app.py` and `core/llm.py` are from different versions. Use matching copies, delete `core/__pycache__` and restart. |
| Old UI still shows | Save the files, stop Streamlit with Ctrl+C, run it again and hard refresh the browser with Ctrl+Shift+R. |
| Theme icon error | Run `pip install -U streamlit`. |
| PDF text is empty or garbled | The PDF may be a scanned image. Use a text-based PDF. |
| Model not found | Check the exact model ID on the Groq console and set `GROQ_MODEL`. |

## Tech stack

Python · Streamlit · scikit-learn (TF-IDF, cosine) · pypdf · Groq API (via the `openai` SDK) · python-dotenv
