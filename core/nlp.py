"""NLP: preprocessing, TF-IDF, cosine similarity, keyword match/missing, RAG retrieval."""
import html
import re
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Words that appear in many JDs but say nothing about skills.
EXTRA_STOP = {
    "experience", "years", "year", "work", "working", "team", "teams", "strong",
    "ability", "skills", "skill", "role", "job", "candidate", "required",
    "preferred", "responsibilities", "requirements", "including", "etc", "good",
    "knowledge", "understanding", "must", "will", "looking", "join", "company",
    "opportunity", "excellent", "plus", "using", "use", "new", "also", "within",
    "across", "based", "like", "able", "well", "help", "build", "support",
}
STOP = set(ENGLISH_STOP_WORDS) | EXTRA_STOP

# Small boost list so important tech terms are never lost. Add your own!
SKILLS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "sql",
    "mysql", "postgresql", "mongodb", "redis", "aws", "azure", "gcp", "docker",
    "kubernetes", "git", "linux", "react", "node", "django", "flask", "fastapi",
    "streamlit", "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
    "keras", "nlp", "llm", "rag", "machine learning", "deep learning",
    "data analysis", "data science", "data visualization", "tableau", "power bi",
    "excel", "spark", "hadoop", "airflow", "etl", "rest api", "api", "ci/cd",
    "jenkins", "agile", "scrum", "html", "css", "opencv", "computer vision",
    "transformers", "langchain", "statistics", "communication", "leadership",
    "problem solving", "microservices", "devops", "testing", "unit testing",
}


def preprocess(text: str) -> str:
    """lowercase -> keep tech chars (+ # . / -) -> normalise spaces."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#./\- \n]", " ", text)
    text = re.sub(r"(?<![a-z0-9])[./\-]+|[./\-]+(?![a-z0-9+#])", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _stem(word: str) -> str:
    """Very light stemmer so 'models' == 'model', 'building' == 'build'."""
    for suf in ("ing", "ers", "er", "es", "s", "ed"):
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            return word[: -len(suf)]
    return word


def _norm_tokens(text: str) -> list[str]:
    return [_stem(t) for t in preprocess(text).split()]


def tokenizer(text: str) -> list[str]:
    toks = [t for t in text.split() if t not in STOP and len(t) > 1]
    return [_stem(t) for t in toks]


@dataclass
class MatchResult:
    cosine_score: float = 0.0
    keyword_score: float = 0.0
    final_score: float = 0.0
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def _skill_hits(clean: str) -> set[str]:
    return {s for s in SKILLS if re.search(rf"(?<![a-z0-9]){re.escape(s)}(?![a-z0-9])", clean)}


def extract_jd_keywords(jd: str, resume: str, top_n: int = 30) -> list[str]:
    """Keywords = known skill phrases in the JD + top TF-IDF single words (original form)."""
    jd_c, res_c = preprocess(jd), preprocess(resume)
    skills = sorted(_skill_hits(jd_c))
    skill_words = {w for s in skills for w in s.split()}

    vec = TfidfVectorizer(tokenizer=lambda t: [w for w in t.split() if w not in STOP and len(w) > 1
                                               and not w.isdigit()],
                          token_pattern=None, lowercase=False, sublinear_tf=True)
    X = vec.fit_transform([jd_c, res_c])
    terms = vec.get_feature_names_out()
    weights = X[0].toarray().ravel()
    ranked = [terms[i] for i in weights.argsort()[::-1] if weights[i] > 0]

    keywords, seen = list(skills), {" ".join(_norm_tokens(s)) for s in skills}
    for t in ranked:
        if len(keywords) >= top_n:
            break
        key = " ".join(_norm_tokens(t))
        if t in skill_words or key in seen:
            continue
        seen.add(key)
        keywords.append(t)
    return keywords


def analyze(resume: str, jd: str, top_n: int = 30) -> MatchResult:
    """Compute TF-IDF cosine similarity + keyword coverage."""
    if not resume.strip() or not jd.strip():
        return MatchResult()
    jd_c, res_c = preprocess(jd), preprocess(resume)

    vec = TfidfVectorizer(tokenizer=tokenizer, token_pattern=None, lowercase=False,
                          ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform([res_c, jd_c])
    cos = float(cosine_similarity(X[0], X[1])[0][0])

    keywords = extract_jd_keywords(jd, resume, top_n)
    res_stems = " " + " ".join(_norm_tokens(resume)) + " "
    matched, missing = [], []
    for k in keywords:
        key = " ".join(_norm_tokens(k))
        (matched if f" {key} " in res_stems else missing).append(k)
    kw = len(matched) / max(len(keywords), 1)
    cos_scaled = min(cos * 2.0, 1.0)  # short docs give low cosine, so stretch it
    final = 0.4 * cos_scaled + 0.6 * kw
    return MatchResult(round(cos * 100, 1), round(kw * 100, 1), round(final * 100, 1),
                       matched, missing)


def highlight_html(text: str, matched: list[str]) -> str:
    """Return HTML of the text with matched keywords wrapped in <mark>."""
    safe = html.escape(text)
    words = sorted({w for k in matched for w in k.split()}, key=len, reverse=True)
    if not words:
        return safe.replace("\n", "<br>")
    pattern = re.compile(r"\b(" + "|".join(re.escape(w) + r"\w{0,3}" for w in words) + r")\b",
                         re.IGNORECASE)
    safe = pattern.sub(r'<mark class="kw-hit">\1</mark>', safe)
    return safe.replace("\n", "<br>")


# ----------------------------------------------------------------- RAG
def chunk_resume(resume: str, max_chars: int = 500) -> list[str]:
    """Split resume into small chunks (by blank line / bullet) for retrieval."""
    parts = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[-•*] )", resume) if p.strip()]
    chunks, buf = [], ""
    for p in parts:
        if len(buf) + len(p) < max_chars:
            buf = f"{buf}\n{p}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def retrieve(resume: str, jd: str, k: int = 6) -> list[str]:
    """RAG retrieval: the k resume chunks most similar to the JD (TF-IDF cosine)."""
    chunks = chunk_resume(resume)
    if len(chunks) <= k:
        return chunks
    vec = TfidfVectorizer(tokenizer=tokenizer, token_pattern=None, lowercase=False)
    X = vec.fit_transform([preprocess(c) for c in chunks] + [preprocess(jd)])
    sims = cosine_similarity(X[-1], X[:-1]).ravel()
    top = sims.argsort()[::-1][:k]
    return [chunks[i] for i in sorted(top)]  # keep original order
