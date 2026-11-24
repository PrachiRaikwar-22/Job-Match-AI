"""LLM wrapper: Groq (OpenAI-compatible API). Key comes from .env."""
import os

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM = (
    "You are an honest career coach and ATS specialist. You NEVER invent or exaggerate. "
    "You only suggest adding a skill or experience to a resume if the candidate's own "
    "resume already shows related or adjacent work that supports it. If there is no such "
    "evidence, you say it is a real gap to learn, not something to add."
)


def build_suggestion_prompt(resume_chunks: list[str], full_resume: str, jd: str,
                            matched: list[str], missing: list[str]) -> str:
    context = "\n---\n".join(resume_chunks)
    return f"""Compare this resume with the job description and give suggestions.

JOB DESCRIPTION:
{jd}

MOST RELEVANT PARTS OF THE RESUME (retrieved):
{context}

FULL RESUME:
{full_resume}

KEYWORDS ALREADY MATCHED: {', '.join(matched) or 'none'}
KEYWORDS MISSING FROM RESUME: {', '.join(missing) or 'none'}

Write the answer in Markdown with exactly these sections:

## ✅ You can add these (you already have related experience)
For each missing keyword where the resume shows nearby / related knowledge:
- **keyword** - Evidence: quote or paraphrase the part of the resume that supports it.
  Suggested wording: one truthful bullet that re-frames that real work using the keyword.

## 📚 Real gaps (learn these)
For each missing keyword with NO supporting evidence: the keyword, why the JD needs it,
and one small project or course idea to gain it honestly. Do NOT suggest putting it on the resume yet.

## ✏️ Other improvements
Max 5 bullets on structure, action verbs, quantifying impact, or ordering sections for this JD.

Ignore generic words that are not skills (e.g. 'need', 'building'). Be concise."""


def generate(prompt: str) -> str:
    """Call Groq. The API key is read from the backend (.env), never from the UI."""
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        raise ValueError("GROQ_API_KEY is missing. Add it to the .env file and restart.")
    from openai import OpenAI  # Groq is OpenAI-compatible
    client = OpenAI(api_key=key, base_url=GROQ_BASE_URL)
    resp = client.chat.completions.create(
        model=GROQ_MODEL, temperature=0.4,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": prompt}])
    return resp.choices[0].message.content