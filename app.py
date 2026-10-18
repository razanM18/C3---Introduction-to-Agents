"""CedarSun Energy business assistant (C3). Generated from business_agent.ipynb.
Run:  python app.py   (needs .env with your API key)
"""


import os, re, json, math, time, datetime
from collections import deque, namedtuple
from pathlib import Path
from dotenv import load_dotenv

try:                                   # only needed when a Space runs on ZeroGPU hardware; harmless elsewhere
    import spaces

    @spaces.GPU
    def _gpu_placeholder():
        return None
except Exception:
    pass

try:
    BASE_DIR = Path(__file__).resolve().parent      # running as app.py
except NameError:
    BASE_DIR = Path.cwd()                            # running as a notebook
load_dotenv(BASE_DIR / ".env")

PROVIDERS = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "key_name": "GEMINI_API_KEY", "default_model": "gemini-3.8-flash"},
    "openai": {"base_url": None,
               "key_name": "OPENAI_API_KEY", "default_model": "gpt-4o-mini"},
    # Free options: Groq (free API key at console.groq.com) and Ollama (runs on your own PC, no key at all).
    "groq": {"base_url": "https://api.groq.com/openai/v1",
             "key_name": "GROQ_API_KEY", "default_model": "openai/gpt-oss-120b"},
    "cerebras": {"base_url": "https://api.cerebras.ai/v1",
                 "key_name": "CEREBRAS_API_KEY", "default_model": "gpt-oss-120b"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                   "key_name": "OPENROUTER_API_KEY", "default_model": "poolside/laguna-s-2.1:free"},
    # Second keys (a separate quota each): used as alternatives when the first key is rate-limited.
    "gemini2": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "key_name": "GEMINI_API_KEY2", "alt_key_names": ["GEMINI_API_KEY_2"], "default_model": "gemini-3.5-flash-lite"},
    "gemini3": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "key_name": "GEMINI_API_KEY3", "alt_key_names": ["GEMINI_API_KEY_3"], "default_model": "gemini-3.5-flash-lite"},
    "openai2": {"base_url": None, "key_name": "OPENAI_API_KEY2", "alt_key_names": ["OPENAI_API_KEY_2"], "default_model": "gpt-4o-mini"},
    "ollama": {"base_url": "http://localhost:11434/v1",
               "key_name": None, "default_model": "llama3.1"},
}
PROVIDER = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
if PROVIDER not in PROVIDERS:
    raise ValueError(f"LLM_PROVIDER must be one of {list(PROVIDERS)}, got {PROVIDER!r}")
MODEL = os.getenv("LLM_MODEL") or PROVIDERS[PROVIDER]["default_model"]

AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.4"))   # warm but controlled
ROUTER_USE_LLM = os.getenv("ROUTER_USE_LLM", "on").lower() not in ("off", "0", "false")
REFLECTION_ENABLED = os.getenv("REFLECTION", "on").lower() not in ("off", "0", "false")
VERBOSE = os.getenv("VERBOSE", "1") != "0"

LOG_DIR = Path(os.getenv("LOG_DIR", str(BASE_DIR / "logs")))
LOG_DIR.mkdir(parents=True, exist_ok=True)


def trace(tag, message):
    """Print a visible trace line, e.g. [router] lead (matched 'quote')."""
    if VERBOSE:
        print(f"[{tag}] {message}")


def get_secret(name):
    """Read a key from the environment/.env, then from Colab Secrets. Placeholders count as missing."""
    value = os.getenv(name)
    if not value:
        try:
            from google.colab import userdata          # only exists in Colab
            value = userdata.get(name)
        except Exception:
            value = None
    if value and (value.lower().startswith("your") or "paste" in value.lower()):
        return None
    return value


def provider_key(cfg):
    """The API key for a provider entry, trying its alternative variable names too."""
    for name in [cfg["key_name"]] + cfg.get("alt_key_names", []):
        value = get_secret(name) if name else None
        if value:
            return value
    return None


_client = None


def get_client():
    """Create the OpenAI-compatible client once. Raises a clear error if the key is missing."""
    global _client
    if _client is None:
        cfg = PROVIDERS[PROVIDER]
        key = provider_key(cfg) if cfg["key_name"] else "ollama"   # Ollama needs no real key
        if not key:
            raise RuntimeError(f"No API key found. Put {cfg['key_name']}=... in the .env file "
                               f"(or add it to Colab Secrets), then re-run.")
        from openai import OpenAI
        _client = OpenAI(api_key=key, base_url=cfg["base_url"], timeout=25, max_retries=0 if os.getenv("LLM_FALLBACKS") else 3)   # with fallbacks, fail over at once instead of waiting
    return _client


def _parse_fallbacks(text):
    """'groq,gemini:gemini-3.7-flash' -> [('groq', None), ('gemini', 'gemini-3.7-flash')]. A provider may repeat with different models."""
    out = []
    for item in (text or "").split(","):
        name, _, model = item.strip().partition(":")
        name = name.lower()
        if name in PROVIDERS and not (name == PROVIDER and not model):
            out.append((name, model.strip() or None))
    return out


FALLBACKS = _parse_fallbacks(os.getenv("LLM_FALLBACKS", ""))
# ---------------------------------------------------------------- KEY POOL (several keys, cooldowns, friendly busy message)
RPM_LIMIT = int(os.getenv("RPM_LIMIT", "0"))                  # requests per minute per key and model; 0 = off. Free Gemini often allows 5.
QUOTA_MAX_WAIT = float(os.getenv("QUOTA_MAX_WAIT", "8"))      # longest pause (seconds) before a reply while a quota window reopens
MIN_QUOTA_COOLDOWN, MAX_QUOTA_COOLDOWN, DAILY_COOLDOWN = 15, 120, 3 * 3600

Endpoint = namedtuple("Endpoint", "provider model idx key")   # one key on one model: quotas are counted per project and model
Endpoint.id = property(lambda self: (self.provider, self.model, self.idx))
Endpoint.label = property(lambda self: f"{self.provider}#{self.idx + 1}:{self.model}")   # never contains the key itself

_clock, _sleep = time.time, time.sleep      # indirection so the tests can run on a fake clock
_clients = {}       # (provider, key) -> OpenAI client
_cooldown = {}      # endpoint id -> time when it may be used again
_cooldown_kind = {} # endpoint id -> why it is resting: quota / auth / missing / outage
_calls = {}         # endpoint id -> timestamps of recent requests (for RPM_LIMIT)


class QuotaBusy(Exception):
    """Every configured key is rate-limited right now; `seconds` is how long until the first one reopens."""
    def __init__(self, seconds):
        self.seconds = max(1, int(math.ceil(seconds)))
        super().__init__(f"all keys are rate-limited; the first one reopens in about {self.seconds}s")


def provider_keys(cfg):
    """Every API key configured for a provider entry: its main variable, its alternative names, and a comma-separated
    plural variable (GEMINI_API_KEY -> GEMINI_API_KEYS). Duplicates are dropped."""
    if not cfg["key_name"]:
        return ["ollama"]                       # Ollama needs no real key
    keys = []
    names = [cfg["key_name"]] + cfg.get("alt_key_names", [])
    names += [f"{cfg['key_name']}_{n}" for n in range(2, 10)]          # GEMINI_API_KEY_2 ... GEMINI_API_KEY_9
    for name in names:
        for raw in (get_secret(name), get_secret(name + "S")):
            for key in (raw or "").split(","):
                key = key.strip()
                if key and key not in keys:
                    keys.append(key)
    return keys


def _endpoints():
    """The main provider first, then the fallbacks, each expanded to one endpoint per key."""
    seen, same_key, out = set(), set(), []
    pairs = [(PROVIDER, MODEL)] + [(n, m or PROVIDERS[n]["default_model"]) for n, m in FALLBACKS]
    for name, model in pairs:
        for idx, key in enumerate(provider_keys(PROVIDERS[name])):
            ep = Endpoint(name, model, idx, key)
            # The same key on the same model and server counts once, however many provider names list it
            # (for example the old gemini2 / gemini3 names next to the pool's own GEMINI_API_KEY_2 / _3).
            identity = (PROVIDERS[name]["base_url"], key, model)
            if ep.id not in seen and identity not in same_key:
                seen.add(ep.id)
                same_key.add(identity)
                out.append(ep)
    return out


def _client_for(ep, single):
    cache_key = (ep.provider, ep.key)
    if cache_key not in _clients:
        from openai import OpenAI
        _clients[cache_key] = OpenAI(api_key=ep.key, base_url=PROVIDERS[ep.provider]["base_url"], timeout=25,
                                     max_retries=2 if single else 0)   # with several keys, move on at once instead of retrying
    return _clients[cache_key]


def _retry_seconds(exc):
    """How long the provider says to wait (Gemini puts it in the message; others use the Retry-After header)."""
    text = str(exc)
    m = re.search(r"retry in ([\d.]+)\s*(ms|s)\b", text, re.I)
    if m:
        value = float(m.group(1))
        return value / 1000 if m.group(2).lower() == "ms" else value
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"]?([\d.]+)s", text)
    if m:
        return float(m.group(1))
    try:
        return float(getattr(getattr(exc, "response", None), "headers", {}).get("retry-after", 0))
    except Exception:
        return 0.0


def _error_kind(exc):
    code, low = getattr(exc, "status_code", None), str(exc).lower()
    if code == 429 or "resource_exhausted" in low or "rate limit" in low or "quota" in low:
        return "quota"
    if code in (401, 403) or "api key not valid" in low or "invalid api key" in low:
        return "auth"
    if code == 404 or "is not found" in low or "model_not_found" in low:
        return "missing"
    if (code and code >= 500) or type(exc).__name__ in ("APITimeoutError", "APIConnectionError", "TimeoutError", "ConnectionError"):
        return "outage"
    return "other"          # for example a 400 about a request option: no cooldown, the next endpoint may accept it


def _cooldown_seconds(kind, exc):
    if kind == "quota":
        if re.search(r"per ?day|\btpd\b", str(exc), re.I):
            return DAILY_COOLDOWN                    # a daily limit will not clear for hours
        return min(max(_retry_seconds(exc), MIN_QUOTA_COOLDOWN), MAX_QUOTA_COOLDOWN)
    return {"auth": 3600, "missing": 600, "outage": 30}.get(kind, 0)


def _ready_at(ep, now, reserve=0):
    """When this endpoint may be used again: its cooldown, or the moment its per-minute window has room."""
    t = _cooldown.get(ep.id, 0.0)
    if RPM_LIMIT:
        q = _calls.setdefault(ep.id, deque())
        while q and now - q[0] >= 60:
            q.popleft()
        if len(q) >= max(RPM_LIMIT - reserve, 1):
            t = max(t, q[0] + 60)
    return t


def _blocked_by_quota(ep, now):
    """True when the endpoint is only waiting for a rate-limit window (not a bad key, a missing model or an outage)."""
    kind = _cooldown_kind.get(ep.id) if _cooldown.get(ep.id, 0.0) > now else None
    return kind in (None, "quota")


def quota_status():
    """One line for the logs, e.g. '2 of 3 keys ready, next one frees up in 12s'. Never shows a key."""
    now, eps = _clock(), _endpoints()
    resting = [_ready_at(e, now) - now for e in eps if _ready_at(e, now) > now]
    line = f"{len(eps) - len(resting)} of {len(eps)} keys ready"
    return line + (f", next one frees up in {min(resting):.0f}s" if resting and len(resting) == len(eps) else "")


def chat(optional=False, **kwargs):
    """chat.completions.create with a pool of keys.
    - Tries the main provider's keys first, then the fallbacks; a key that hit its limit rests until the provider says it reopens.
    - If every key is resting but one reopens within QUOTA_MAX_WAIT seconds, waits for it; otherwise raises QuotaBusy.
    - optional=True is for helper calls (router, reflection, language fix): they never wait, they skip when quota is tight,
      and with RPM_LIMIT they leave room for the main reply."""
    eps = _endpoints()
    if not eps:
        raise RuntimeError("No LLM provider is configured: add an API key to .env")
    reserve, budget, first_error, tried = (2 if optional else 0), (0.0 if optional else QUOTA_MAX_WAIT), None, set()
    for _round in range(6):
        now = _clock()
        for ep in [e for e in eps if e.id not in tried and _ready_at(e, now, reserve) <= now]:
            tried.add(ep.id)
            if first_error is not None:
                trace("keys", f"trying {ep.label}")
            _calls.setdefault(ep.id, deque()).append(_clock())
            try:
                resp = _client_for(ep, single=len(eps) == 1).chat.completions.create(model=ep.model, **kwargs)
                _cooldown.pop(ep.id, None)
                return resp
            except Exception as exc:
                first_error = first_error or exc
                kind = _error_kind(exc)
                seconds = _cooldown_seconds(kind, exc)
                if seconds:
                    _cooldown[ep.id], _cooldown_kind[ep.id] = _clock() + seconds, kind
                trace("keys", f"{ep.label} failed ({kind}, {type(exc).__name__})"
                              + (f"; resting {seconds:.0f}s" if seconds else "") + f". {quota_status()}")
        now = _clock()
        waits = [_ready_at(e, now, reserve) - now for e in eps
                 if _ready_at(e, now, reserve) > now and _blocked_by_quota(e, now)]
        if not waits:                        # nothing is merely rate-limited: a real error (bad key, bad request, outage)
            raise first_error or RuntimeError("No LLM provider is available")
        soonest = min(waits)
        if soonest > budget:
            raise QuotaBusy(soonest)
        trace("keys", f"all keys busy; waiting {soonest:.1f}s for one to reopen")
        _sleep(soonest + 0.2)
        budget -= soonest
        tried.clear()
    raise QuotaBusy(MIN_QUOTA_COOLDOWN)


BUSY_TEXT = {
    "english": ("I'm getting a lot of questions at the moment. Please send your message again in about {when}, "
                "or email hello@cedarsun-energy.example and the team will reply."),
    "french": ("Je reçois beaucoup de questions en ce moment. Pouvez-vous renvoyer votre message dans environ {when} ? "
               "Vous pouvez aussi écrire à hello@cedarsun-energy.example."),
    "arabic": ("عندي أسئلة كتير هلأ. ابعتلي رسالتك مرة تانية بعد حوالي {when}، "
               "أو راسل الفريق على hello@cedarsun-energy.example."),
    "arabizi": ("3andi ktir as2ile hala2. Ib3atli resaltak marra tenye ba3d hawali {when}, "
                "aw ba3at email la hello@cedarsun-energy.example."),
}
WAIT_UNITS = {   # seconds, minutes, hours-or-more
    "english": ("{n} seconds", "{n} minutes", "a few hours"),
    "french": ("{n} secondes", "{n} minutes", "quelques heures"),
    "arabic": ("{n} ثانية", "{n} دقيقة", "كم ساعة"),
    "arabizi": ("{n} sanye", "{n} d2i2a", "kam sa3a"),
}


def busy_message(message, history, seconds):
    """The 'too many requests' reply, in the customer's language, with an honest wait time."""
    lang = expected_language(message, history)
    sec_unit, min_unit, long_unit = WAIT_UNITS[lang]
    if seconds >= 3600:
        when = long_unit
    elif seconds >= 90:
        when = min_unit.format(n=math.ceil(seconds / 60))
    else:
        when = sec_unit.format(n=int(math.ceil(seconds / 5.0) * 5))
    return BUSY_TEXT[lang].format(when=when)



trace("config", f"provider={PROVIDER} model={MODEL} reflection={REFLECTION_ENABLED} router_llm={ROUTER_USE_LLM}")
trace("keys", f"{len(_endpoints())} key/model endpoint(s): " + ", ".join(e.label for e in _endpoints()) + f"; RPM_LIMIT={RPM_LIMIT or 'off'}")


def load_knowledge():
    me = BASE_DIR / "me"
    summary = (me / "business_summary.txt").read_text(encoding="utf-8").strip()
    pdf_text = ""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(me / "about_business.pdf"))
        pdf_text = "\n".join((page.extract_text() or "") for page in reader.pages)
        pdf_text = re.sub(r"[ \t]+", " ", pdf_text)
        pdf_text = re.sub(r"\n{3,}", "\n\n", pdf_text).strip()
    except Exception as exc:
        trace("knowledge", f"could not read about_business.pdf ({exc}); using the summary only")
    return summary, pdf_text


BUSINESS_SUMMARY, BUSINESS_PDF = load_knowledge()
trace("knowledge", f"summary: {len(BUSINESS_SUMMARY.split())} words, pdf: {len(BUSINESS_PDF.split())} words")


# ---------------------------------------------------------------- PERSONA (who she is)
PERSONA = """You are Zeina, a solar advisor at CedarSun Energy, a residential solar company in Beirut, Lebanon.

Who you are: a warm, down-to-earth energy engineer who genuinely enjoys helping families get reliable electricity.
You know solar well, but you never talk down to anyone.

How you sound (like a real person texting a customer on WhatsApp, not like a company bot):
- Short, natural messages. Usually two or three sentences. React to what the customer actually said before
  answering ("Ah, a flat in Achrafieh, nice", "Fair question", "Ugh, the cuts are exhausting").
- Everyday words, contractions, a little warmth and the occasional light touch of humour. No corporate phrases,
  no "I'd be happy to assist", no repeating the customer's question back, no bullet points or headings.
- Vary how you open and close messages. Do not start every message with "Ahlan" and do not always finish with
  the same kind of question.
- Confident about what you know, relaxed and honest about what you do not ("let me check with the team").
- Language: always reply in the language and script the customer is using (English, French, Arabic or Arabizi) and
  never switch on your own. A language instruction is added to every turn: follow it.
- Arabic and Arabizi must be Lebanese dialect, the way people really talk in Beirut, never formal Modern Standard
  Arabic. Use words like: shu, kifak/kifik, hala2/halla, ktir, mnih, bade/badde, 3am, eh, la2, tekram, inshallah,
  yalla, ma fi mshekle, 7ader, tfaddal. In Arabic script: شو، كيفك، هلأ، كتير، منيح، بدّي، عم، إي، لأ، تكرم، يلا،
  ما في مشكلة، حاضر. Avoid stiff phrases like "يمكنني مساعدتك" or "ما هو"; say "شو بتحب تعرف" or "شو بدّك" instead.
  Lebanese Arabizi uses 3 for ع, 7 for ح, 2 for ء, 5 for خ, 8 for غ, e.g. "ahla w sahla, shu badak ta3ref?".
  French is casual and warm, the way Lebanese people speak it.
"""

# ---------------------------------------------------------------- INSTRUCTIONS (what she must do)
INSTRUCTIONS = """\
INPUTS you receive each turn
- The customer's latest message and the recent chat history.
- An INTENT HINT from the router (general, lead or feedback) explaining how to handle this turn.
- Tool results, returned as JSON, after you call a tool.

OUTPUT you must return
- Plain conversational text only: no headings, no tables, no markdown lists.
- Keep it short, like a chat message: about 40 to 80 words, up to 120 only when the customer asks for detail.
- Move the conversation forward with one question or next step when it feels natural; do not force a closing
  question onto every message, and never ask more than one.
- Never open with a greeting after the first reply, and never repeat the same disclaimer wording twice in a row.
- Give numbers with units (kWh, kWp, kW, panels).

RULES (each rule has a single purpose)
R1. Ground every fact about CedarSun (services, packages, warranty, timelines, team, policies) in the KNOWLEDGE BASE below.
R2. If the answer is not in the knowledge base, say so honestly ("that's not something I have details on") and call
    record_feedback with the customer's question. Never claim CedarSun does or does not offer something unless the
    knowledge base says so explicitly. Quote phone numbers and emails exactly as the knowledge base gives them: use the
    support line for faults and the general contact for everything else.
R3. Never state, estimate or hint at prices, discounts, savings, payback or financing. Explain that every price is a written
    quotation after the free site survey, and call record_feedback with the price question.
R4. Never give regulatory, legal, grid-export, net-metering, tax or insurance information. Call record_feedback instead.
R5. Never do sizing arithmetic yourself. Call estimate_system_size and report exactly the numbers it returns.
R6. Present every sizing result as approximate and subject to the free site survey.
R7. For sizing, once the customer has named their appliances (and ideally the area), you MUST call estimate_system_size
    in that same turn, even if the customer writes in Arabic, Arabizi or French: translate the appliance names to English
    (براد = refrigerator, تكييف = air conditioner, تلفزيون = television) when calling the tool. If watts or hours are missing, leave them out and let the tool use typical values, and mention that typical
    values were used. Never ask for hours or watts a second time. Ask only for what is truly missing, one or two
    questions at a time: home type or size, main appliances, whole home or essentials only, and the area.
R8. Never give electrical installation or repair instructions. For faults, point to the support line. For smoke, a burning
    smell, sparking or a damaged battery, tell the customer to switch the system off at the isolators only if it is safe,
    keep away, call the CedarSun support line and call emergency services if there is a fire.
R9. Encourage leads: after a useful answer about sizing or services, invite the customer to leave a name and email for a
    free site survey. Never end a lead conversation without asking for an email.
R10. Call record_customer_interest only when you have a valid email and a name. Put a one-line summary of the customer's
     needs in "message". Never invent contact details.
R11. Stay on topic (solar, batteries, energy use, CedarSun). Politely steer other subjects back in one friendly sentence and do
     not call record_feedback for them.
R12. Be honest about what you are: if a customer sincerely asks whether they are talking to a person or an AI, say plainly
     that you are CedarSun's AI assistant, then carry on helping. Never claim to be human or to have a body, home or family.

WORKED EXAMPLE
Customer: "How many panels do I need? I live in a flat in Beirut with a fridge, lights, a TV and one AC."
Zeina (calls estimate_system_size with those appliances and area "Beirut", then replies):
"For that setup you are looking at roughly 5 panels, around 2.25 kWp. Take it as a ballpark though, shading and your
roof direction can change it, and the free site survey is where we confirm it. Are you hoping to run the whole flat
during cuts, or just the essentials?"
"""


# ---------------------------------------------------------------- SYSTEM PROMPT (persona + instructions + knowledge)
PDF_MAX_WORDS = int(os.getenv("PDF_MAX_WORDS", "0"))   # 0 = send the whole PDF; e.g. 1200 for small-quota free tiers


def select_pdf_text(query, max_words):
    """Keep only the PDF paragraphs that best match the customer's words (for providers with small token limits)."""
    if not max_words or len(BUSINESS_PDF.split()) <= max_words:
        return BUSINESS_PDF
    paras = [p.strip() for p in re.split(r"\n\s*\n", BUSINESS_PDF) if p.strip()]
    q = set(re.findall(r"[a-z0-9]{3,}", (query or "").lower()))
    scored = [(len(q & set(re.findall(r"[a-z0-9]{3,}", p.lower()))), i, p) for i, p in enumerate(paras)]
    chosen, used = [], 0
    for score, i, p in sorted(scored, key=lambda t: (-t[0], t[1])):
        n = len(p.split())
        if used + n <= max_words:
            chosen.append((i, p)); used += n
    return "\n\n".join(p for _, p in sorted(chosen))


def build_system_prompt(query=""):
    return (
        "=== PERSONA ===\n" + PERSONA.strip() + "\n\n"
        "=== INSTRUCTIONS ===\n" + INSTRUCTIONS.strip() + "\n\n"
        "=== KNOWLEDGE BASE: BUSINESS SUMMARY ===\n" + BUSINESS_SUMMARY + "\n\n"
        "=== KNOWLEDGE BASE: ABOUT_BUSINESS.PDF ===\n" + (select_pdf_text(query, PDF_MAX_WORDS) or "(not available)")
    )


SYSTEM_PROMPT = build_system_prompt()
trace("prompt", f"system prompt is about {len(SYSTEM_PROMPT.split())} words")


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _log(filename, record):
    record = {"timestamp": _now(), **record}
    with open(LOG_DIR / filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")


def record_customer_interest(email: str, name: str, message: str) -> dict:
    """Tool 1: save a customer lead."""
    email, name, message = (email or "").strip(), (name or "").strip(), (message or "").strip()
    if not EMAIL_RE.match(email):
        return {"status": "error", "error": "invalid_email",
                "detail": "The email address looks invalid. Ask the customer to confirm it."}
    if not name:
        return {"status": "error", "error": "missing_name", "detail": "Ask the customer for their name."}
    _log("leads.jsonl", {"name": name, "email": email, "message": message})
    trace("tool", f"record_customer_interest -> lead saved: {name} <{email}> | {message}")
    return {"status": "ok", "saved": {"name": name, "email": email},
            "next_step": "A CedarSun specialist will contact the customer within 1 business day."}


def record_feedback(question: str) -> dict:
    """Tool 2: log feedback or a question the agent cannot answer."""
    question = (question or "").strip()
    if not question:
        return {"status": "error", "error": "empty_question"}
    _log("feedback.jsonl", {"question": question})
    trace("tool", f"record_feedback -> logged: {question}")
    return {"status": "ok", "logged": question,
            "next_step": "The question was passed to the CedarSun team for a verified answer."}


# ---- Tool 3: sizing estimator, using the planning assumptions written in about_business.pdf (Section 7)
PANEL_KW = 0.45             # 450 W panels
PERFORMANCE_FACTOR = 0.80
DOD, DISCHARGE_EFF = 0.90, 0.90
BATTERY_MODULE_KWH = 5.12
DEFAULT_BACKUP_SHARE = 0.40   # battery sized for 40% of daily energy when no backup target is given
DIVERSITY = 0.70              # about 70% of connected load runs at once
INVERTER_SIZES_KW = [3, 5, 8]

AREA_TABLE = [  # (keywords, region, planning peak-sun hours)
    (["beirut", "jbeil", "byblos", "batroun", "tripoli", "saida", "sidon"], "Coast", 5.0),
    (["metn", "keserwan", "baabda", "aley", "chouf", "mount lebanon"], "Mount Lebanon", 5.2),
    (["zahle", "zahlé", "bekaa", "beqaa"], "Bekaa", 5.5),
]
# (name keywords, watts, hours per day, label); order matters, first match wins
TYPICAL = [
    (["water heater", "boiler", "geyser"], 2000, 1.5, "Electric water heater"),
    (["refrigerator", "fridge", "freezer"], 150, 8, "Refrigerator"),
    (["air con", "a/c", "aircon", "conditioner", "split unit", r"ac"], 1000, 5, "Air conditioner"),
    (["tv", "television"], 100, 4, "Television"),
    (["light", "lamp"], 100, 5, "Lighting"),
    (["washing", "washer", "laundry"], 500, 1, "Washing machine"),
]
ENGINEER_REVIEW = re.compile(r"\b(oven|cook|stove|water heater|boiler|geyser|kiln|welding)\b", re.I)
PACKAGES = [  # (name, panels, inverter kW, battery modules)
    ("Essential Backup", 4, 3, 1),
    ("Smart Home Hybrid", 8, 5, 2),
    ("High-Demand Home", 14, 8, 3),
]


def _typical(name):
    n = name.lower()
    for keys, watts, hours, label in TYPICAL:
        for k in keys:
            if (k == "ac" and re.search(r"\bac\b", n)) or (k != "ac" and re.search(r"\b" + re.escape(k), n)):
                return watts, hours, label
    return None


def _sun_hours(area):
    key = (area or "").strip().lower()
    for words, region, hours in AREA_TABLE:
        if any(w in key for w in words):
            return hours, region, None
    note = ("Area not recognised or not given, so the default 5.0 peak-sun hours was used; "
            "it may also be outside CedarSun's service areas.")
    return 5.0, "Unknown (default)", note


def estimate_system_size(appliances: list, area: str = "", backup_kwh: float = None) -> dict:
    """Tool 3: rough, approximate system size from appliances, hours and area."""
    rows, assumed, unknown, engineer = [], [], [], False
    has_ac = False
    for a in appliances or []:
        name = str(a.get("name", "")).strip() or "appliance"
        qty = int(a.get("quantity") or 1)
        watts, hours = a.get("watts"), a.get("hours_per_day")
        typ = _typical(name)
        if typ:
            label = typ[2]
            if watts is None:
                watts = typ[0]; assumed.append(f"{name}: assumed {typ[0]} W")
            if hours is None:
                hours = typ[1]; assumed.append(f"{name}: assumed {typ[1]} h/day")
            has_ac = has_ac or label == "Air conditioner"
        if watts is None or hours is None:
            unknown.append(name)
            continue
        if ENGINEER_REVIEW.search(name):
            engineer = True
        rows.append({"appliance": name, "quantity": qty, "watts": float(watts), "hours_per_day": float(hours),
                     "kwh_per_day": round(float(watts) * float(hours) * qty / 1000, 2)})
    if not rows:
        return {"status": "needs_input",
                "detail": "No appliance had usable power and hours. Ask the customer which appliances they use "
                          "and roughly how many hours a day.", "unknown_appliances": unknown}

    daily_kwh = sum(r["kwh_per_day"] for r in rows)
    sun, region, area_note = _sun_hours(area)
    array_kw = daily_kwh / (sun * PERFORMANCE_FACTOR)
    panels = max(1, math.ceil(round(array_kw / PANEL_KW, 6)))
    kwp = panels * PANEL_KW

    target_kwh = float(backup_kwh) if backup_kwh else DEFAULT_BACKUP_SHARE * daily_kwh
    nominal_kwh = target_kwh / (DOD * DISCHARGE_EFF)
    modules = max(1, math.ceil(round(nominal_kwh / BATTERY_MODULE_KWH, 6)))

    connected_w = sum(r["watts"] * r["quantity"] for r in rows)
    needed_kw = connected_w * DIVERSITY / 1000
    needed_kw = max(needed_kw, 5 if has_ac else 3)
    inverter = next((s for s in INVERTER_SIZES_KW if s >= needed_kw), None)
    custom = inverter is None or engineer
    inverter = inverter or 8

    match = next((p for p in PACKAGES if p[1] >= panels and p[2] >= inverter and p[3] >= modules), None)
    result = {
        "status": "ok",
        "appliances": rows,
        "assumed_values": assumed,
        "unknown_appliances_excluded": unknown,
        "area_used": region, "peak_sun_hours": sun,
        "daily_energy_kwh": round(daily_kwh, 2),
        "required_array_kw": round(array_kw, 2),
        "panels_450w": panels, "array_kwp": round(kwp, 2),
        "expected_daily_production_kwh": round(kwp * sun * PERFORMANCE_FACTOR, 1),
        "battery_target_kwh": round(target_kwh, 2),
        "battery_modules_5_12kwh": modules, "battery_nominal_kwh": round(modules * BATTERY_MODULE_KWH, 2),
        "inverter_kw": inverter,
        "closest_illustrative_package": match[0] if (match and not custom) else "Custom Design (engineer review)",
        "needs_engineer_review": custom,
        "disclaimer": ("Approximate planning estimate only. It ignores shading, roof orientation, season and dust. "
                       "The final design follows the free CedarSun site survey."),
    }
    if area_note:
        result["area_note"] = area_note
    trace("tool", f"estimate_system_size -> {daily_kwh:.1f} kWh/day, {panels} panels ({kwp:.2f} kWp), "
                  f"{modules} battery module(s), {inverter} kW inverter")
    return result


TOOL_DEFINITIONS = [
    {
        "name": "record_customer_interest",
        "description": "Save a customer lead once you have their name and a valid email.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Customer email address"},
                "name": {"type": "string", "description": "Customer name"},
                "message": {"type": "string", "description": "One-line summary of what they need (home, appliances, estimate)"},
            },
            "required": ["email", "name", "message"],
        },
        "output": '{"status": "ok", "saved": {...}, "next_step": "..."} or {"status": "error", "error": "invalid_email"}',
    },
    {
        "name": "record_feedback",
        "description": "Log a complaint, suggestion, or a question you cannot answer from the knowledge base.",
        "inputSchema": {
            "type": "object",
            "properties": {"question": {"type": "string", "description": "The customer's feedback or unanswered question, in their words"}},
            "required": ["question"],
        },
        "output": '{"status": "ok", "logged": "...", "next_step": "..."}',
    },
    {
        "name": "estimate_system_size",
        "description": "Rough solar system size (panels, battery, inverter) from appliances and area. Always use this for sizing; never calculate yourself. Appliance names must be in English.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "appliances": {
                    "type": "array",
                    "description": "Appliances the customer wants to power",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "e.g. fridge, air conditioner, TV, lighting, washing machine"},
                            "watts": {"type": "number", "description": "Running power in watts; omit if unknown"},
                            "hours_per_day": {"type": "number", "description": "Daily hours of use; omit if unknown"},
                            "quantity": {"type": "integer", "description": "How many of this appliance (default 1)"},
                        },
                        "required": ["name"],
                    },
                },
                "area": {"type": "string", "description": "City or region, e.g. Beirut, Metn, Zahle"},
                "backup_kwh": {"type": "number", "description": "Energy the customer wants available from the battery (kWh); omit if unknown"},
            },
            "required": ["appliances"],
        },
        "output": '{"status": "ok", "daily_energy_kwh", "panels_450w", "array_kwp", "battery_modules_5_12kwh", "inverter_kw", "disclaimer", ...}',
    },
]

TOOL_FUNCTIONS = {
    "record_customer_interest": record_customer_interest,
    "record_feedback": record_feedback,
    "estimate_system_size": estimate_system_size,
}


def to_openai_tools(names):
    return [{"type": "function",
             "function": {"name": d["name"], "description": d["description"], "parameters": d["inputSchema"]}}
            for d in TOOL_DEFINITIONS if d["name"] in names]


def run_tool(name, arguments_json):
    """Execute a tool call requested by the model; always return a JSON string (errors included)."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return json.dumps({"status": "error", "error": f"unknown tool {name}"})
    try:
        args = json.loads(arguments_json) if arguments_json else {}
        return json.dumps(func(**args), ensure_ascii=False)
    except Exception as exc:
        trace("tool", f"{name} failed: {exc}")
        return json.dumps({"status": "error", "error": str(exc)})


LEAD_RE = re.compile(
    r"(\bquote\b|\bquotation\b|\bsign me up\b|\bcall me\b|\bcontact me\b|\breach me\b|\bget started\b|"
    r"\b(book|schedule|arrange|request)\w*\b.{0,25}\b(survey|visit|appointment|call)\b|"
    r"\b(come|send someone)\b.{0,20}\b(see|visit|check)\b|\bi('m| am) interested\b|"
    r"\bmy (name|email|number|phone|whatsapp) is\b|\bwhatsapp\b|\bappointment\b)", re.I)
FEEDBACK_RE = re.compile(
    r"(\bcomplain\w*|\bdisappointed\b|\bunhappy\b|\bnot happy\b|\bterrible\b|\bawful\b|\bbad service\b|"
    r"\bsuggest\w*\b|\bfeedback\b|\bnot working\b|\bstopped working\b|\bbroken\b|\bfault\w*\b|\brefund\b|"
    r"\b(price|prices|pricing|cost|costs|discount\w*|financ\w+|install?ments?|loan\w*|payback|incentives?|subsid\w+|"
    r"regulations?|licen[cs]e\w*|permits?|tax(es)?|insurance|legal\w*)\b|"
    r"\bnet[- ]?metering\b|\bgrid export\b|\bexport\w* (power|to the grid)\b|\bsell (power|electricity)\b)", re.I)
INTENTS = ("general", "lead", "feedback")


def route_heuristic(message):
    """Return (intent, matched_text) if a keyword rule decides, otherwise (None, None)."""
    if re.search(r"[^@\s]+@[^@\s]+\.[^@\s]+", message):
        return "lead", "an email address"
    m = LEAD_RE.search(message)
    if m:
        return "lead", m.group(0)
    m = FEEDBACK_RE.search(message)
    if m:
        return "feedback", m.group(0)
    return None, None


ROUTER_PROMPT = """\
Classify the customer's latest message for a solar company's chatbot into exactly one label.
general  - questions about the company, solar, sizing, packages, process, warranty, answers to the advisor's own questions,
           and anything unrelated to solar or the company (small talk, sports, poems): those are simply off-topic
lead     - wants a quote, survey, visit or callback, or is giving name, email or contact details
feedback - a complaint, praise, suggestion, fault report, OR a question company documents cannot answer
           (prices, financing, regulations, net metering or grid export, taxes, insurance, permits)
Reply with one word only: general, lead or feedback."""


def route_llm(message, history):
    last_assistant = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
    user = f"Advisor's previous message: {last_assistant[:400]!r}\nCustomer's latest message: {message!r}"
    resp = chat(
        optional=True, temperature=0,
        messages=[{"role": "system", "content": ROUTER_PROMPT}, {"role": "user", "content": user}])
    text = (resp.choices[0].message.content or "").lower()
    found = [i for i in INTENTS if re.search(rf"\b{i}\b", text)]
    return found[0] if found else "general"


def route_message(message, history=None):
    """Classify a message into general / lead / feedback."""
    history = history or []
    intent, why = route_heuristic(message)
    if intent:
        trace("router", f"{intent} (rule matched: {why!r})")
        return intent
    if ROUTER_USE_LLM:
        try:
            intent = route_llm(message, history)
            trace("router", f"{intent} (LLM classifier)")
            return intent
        except Exception as exc:
            trace("router", f"LLM classifier unavailable ({exc}); defaulting to general")
    else:
        trace("router", "general (default)")
    return "general"


HISTORY_WINDOW = 12
MAX_TOOL_ROUNDS = 5

INTENT_HINTS = {
    "general": ("INTENT HINT: general question. Answer from the knowledge base in your own words. For sizing, collect the "
                "missing details one or two at a time, then call estimate_system_size. If the answer is not in the knowledge "
                "base, call record_feedback. After a helpful answer, gently invite the customer to leave a name and email "
                "for a free site survey."),
    "lead": ("INTENT HINT: lead. The customer wants a quote, a survey or a callback. Collect whatever is missing (name, "
             "email, a short note about their home or appliances), one item at a time. Call record_customer_interest as "
             "soon as you have a name and a valid email, then confirm that a specialist will call within 1 business day. "
             "Do not mention or estimate prices."),
    "feedback": ("INTENT HINT: feedback or unanswered question. Call record_feedback once with the customer's exact words. "
                 "Reply with empathy and say the team will follow up. If it is about price, regulations or grid export, "
                 "explain what you can (for example: quotations are written after the free site survey) without giving "
                 "numbers or legal information. Still answer any part of the message that the knowledge base covers. "
                 "Offer to take their contact details."),
}
INTENT_TOOLS = {
    "general": ["estimate_system_size", "record_feedback"],
    "lead": ["record_customer_interest", "estimate_system_size"],
    "feedback": ["record_feedback", "estimate_system_size"],
}


ARABIZI_WORDS = {"kifak", "kifik", "kifkon", "shu", "shou", "chou", "bade", "badde", "badi", "mnih", "ktir", "kteer", "halla",
                 "hala2", "habibi", "habibti", "marhaba", "merhaba", "shukran", "yalla", "inshallah", "ma3", "3am", "eh",
                 "la2", "lek", "ahla", "ahlan", "tekram", "tfaddal", "w", "fi", "ma", "el", "bel", "3ala", "3andi", "mshekle",
                 "sa3a", "lesh", "leh", "hek", "heik", "aadi", "3adi", "shi", "ya3ni", "ana", "inta", "enta", "7abibi"}
FRENCH_WORDS = {"bonjour", "bonsoir", "salut", "je", "j", "voudrais", "veux", "combien", "panneaux", "panneau", "il", "faut",
                "pour", "mon", "mes", "appartement", "maison", "merci", "vous", "avez", "quel", "quelle", "est", "sont",
                "les", "des", "une", "un", "du", "au", "aux", "besoin", "prix", "garantie", "avec", "sans", "et", "ou",
                "que", "qui", "pas", "oui", "non", "comment", "pourquoi", "installation", "rendez", "devis", "svp", "plait"}


ENGLISH_WORDS = {"the", "you", "your", "are", "is", "and", "to", "my", "for", "with", "what", "how", "do", "does", "can", "i", "it",
                 "we", "of", "in", "on", "that", "this", "me", "have", "tell", "now", "all", "need", "much", "many", "would", "like"}


def detect_language(text):
    """Return 'arabic', 'arabizi', 'french', 'english', or None when the text is too short or neutral to tell."""
    text = text or ""
    if re.search(r"[؀-ۿ]", text):
        return "arabic"
    words = re.findall(r"[a-zA-Z0-9À-ÿ']+", text.lower())
    if not words or re.search(r"@\w", text):
        return None
    digit_words = sum(1 for w in words if re.search(r"[a-z][23578]|[23578][a-z]", w))
    arabizi_hits = sum(1 for w in words if w in ARABIZI_WORDS - {"w", "fi", "ma", "el", "eh", "ana", "ahlan"}) + digit_words
    if arabizi_hits >= 1 and (arabizi_hits >= 2 or len(words) <= 6):
        return "arabizi"
    french_hits = sum(1 for w in words if w in FRENCH_WORDS) + (1 if re.search(r"[éèêàçùôî]", text.lower()) else 0)
    english_hits = sum(1 for w in words if w in ENGLISH_WORDS)
    if french_hits > english_hits and french_hits >= 2 or (french_hits == 1 and len(words) <= 2 and words[0] in {"bonjour", "bonsoir", "salut", "merci"}):
        return "french"
    if len(words) >= 3 or any(w in {"hi", "hello", "hey", "thanks", "thank"} for w in words):
        return "english"
    return None


LANGUAGE_NAMES = {
    "arabic": "Lebanese Arabic (عامية لبنانية) written in Arabic script",
    "arabizi": "Lebanese Arabizi (Lebanese Arabic written in Latin letters and numbers)",
    "french": "French",
    "english": "English",
}


def language_instruction(message, history):
    """Pick the conversation language from the latest customer message that shows one; short replies keep the previous language."""
    user_texts = [message] + [m["content"] for m in reversed(history) if m["role"] == "user"]
    lang = next((l for l in map(detect_language, user_texts) if l), "english")
    return (f"LANGUAGE FOR THIS REPLY: write the whole reply in {LANGUAGE_NAMES[lang]}. Never mix scripts (Arabizi means no Arabic letters at all) and do not switch language unless the "
            f"customer's message clearly switches. Tool results and the knowledge base are in English: translate, do not copy.")


def _content_text(content):
    """Gradio can pass message content as a string or as a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content or "")


def normalize_history(history):
    """Accept Gradio history as messages [{role, content}] or as [[user, bot], ...] pairs."""
    clean = []
    for item in history or []:
        if isinstance(item, dict):
            clean.append({"role": item.get("role", "user"), "content": _content_text(item.get("content"))})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            if item[0]:
                clean.append({"role": "user", "content": _content_text(item[0])})
            if item[1]:
                clean.append({"role": "assistant", "content": _content_text(item[1])})
    return clean[-HISTORY_WINDOW:]


def _assistant_tool_message(msg):
    """Rebuild the assistant message that requested tools, keeping only fields every provider accepts."""
    calls = []
    for tc in msg.tool_calls:
        entry = {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
        extra = getattr(tc, "model_extra", None) or {}
        if "extra_content" in extra:                      # Gemini thought signatures, if present
            entry["extra_content"] = extra["extra_content"]
        calls.append(entry)
    return {"role": "assistant", "content": msg.content or "", "tool_calls": calls}


SIZING_APPLIANCE_RE = re.compile(
    r"(fridge|refrigerator|freezer|\bac\b|air[- ]?con|conditioner|\btv\b|television|lights?|lamps?|washing|washer|boiler|water heater|"
    r"frigo|r[ée]frig|clim|t[ée]l[ée]|lumi[eè]re|machine [àa] laver|"
    r"brad|tele?fizy?on|takyif|kayyef|dawe|ghassale|"
    r"براد|ثلاج|تلفزيون|تلفاز|تكييف|مكيف|إضاءة|غسالة|سخان)", re.I)
SIZING_ASK_RE = re.compile(
    r"(panel|panneau|size|sizing|how many|how much (power|energy)|need|combien|dimension|besoin|kam |kem |bde |bade |badde |"
    r"لوح|ألواح|كم |بدي|بحتاج|احتاج)", re.I)
FORWARD_CLAIM_RE = re.compile(
    r"(passed|logged|forwarded|relayed|noted|recorded|flagged|alerted|shared)\b.{0,60}\b(team|colleagues)|"
    r"(transmis|transf[ée]r[ée]|not[ée]|signal[ée]).{0,60}[ée]quipe|"
    r"(فريق|الفريق)", re.I)


def wants_sizing(message):
    """True when the customer names appliances and asks for a size: the sizing tool must be used."""
    return bool(SIZING_APPLIANCE_RE.search(message) and SIZING_ASK_RE.search(message))


def expected_language(message, history):
    user_texts = [message] + [m["content"] for m in reversed(history) if m["role"] == "user"]
    return next((l for l in map(detect_language, user_texts) if l), "english")


def enforce_language(reply, message, history):
    """Rewrite the reply once if it came back in a different language than the customer's."""
    want, got = expected_language(message, history), detect_language(reply)
    mismatch = (want == "arabizi" and got == "arabic") or (want != "arabizi" and got and got != want)
    if not mismatch:
        return reply
    trace("language", f"reply was {got} but the customer wrote {want}; rewriting")
    try:
        resp = chat(optional=True, temperature=0, messages=[
            {"role": "system", "content": f"Rewrite the message below in {LANGUAGE_NAMES[want]}. Keep the meaning, tone, numbers and "
                                          f"length exactly. Return only the rewritten message."},
            {"role": "user", "content": reply}])
        fixed = (resp.choices[0].message.content or "").strip()
        return fixed or reply
    except Exception as exc:
        trace("language", f"rewrite skipped ({type(exc).__name__})")
        return reply


def run_agent(user_message, history=None):
    """Full pipeline for one customer message. Returns a dict with the reply and a trace of what happened."""
    history = normalize_history(history)
    route = route_message(user_message, history)
    search_text = user_message + " " + " ".join(m["content"] for m in history[-4:] if m["role"] == "user")
    system = (build_system_prompt(search_text) + "\n\n" + INTENT_HINTS[route]
              + "\n\n" + language_instruction(user_message, history))
    messages = ([{"role": "system", "content": system}]
                + history + [{"role": "user", "content": user_message}])
    tools = to_openai_tools(INTENT_TOOLS[route])
    tools_used, tool_results, draft = [], [], None

    force_sizing = route != "feedback" and wants_sizing(user_message)
    for round_no in range(MAX_TOOL_ROUNDS):
        extra = {}
        if force_sizing and round_no == 0:
            trace("safeguard", "sizing request: requiring estimate_system_size on the first step")
            extra["tool_choice"] = {"type": "function", "function": {"name": "estimate_system_size"}}
        try:
            resp = chat(messages=messages, tools=tools, temperature=AGENT_TEMPERATURE, **extra)
        except QuotaBusy:
            raise
        except Exception:
            if not extra:
                raise
            resp = chat(messages=messages, tools=tools, temperature=AGENT_TEMPERATURE)   # provider without forced tools
        msg = resp.choices[0].message
        if not msg.tool_calls:
            draft = (msg.content or "").strip()
            break
        messages.append(_assistant_tool_message(msg))
        for i, tc in enumerate(msg.tool_calls):
            trace("agent", f"model requested tool {tc.function.name}({tc.function.arguments})")
            result = run_tool(tc.function.name, tc.function.arguments)
            tools_used.append(tc.function.name)
            tool_results.append({"tool": tc.function.name, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.id or f"call_{i}", "content": result})
    if draft is None:   # too many tool rounds: force a text answer
        resp = chat(messages=messages, temperature=AGENT_TEMPERATURE)
        draft = (resp.choices[0].message.content or "").strip()

    # Safeguard: a feedback message must always be logged, even if the model forgot to call the tool.
    if route == "feedback" and "record_feedback" not in tools_used:
        trace("safeguard", "model did not log the feedback, logging it now")
        record_feedback(user_message)
        tools_used.append("record_feedback (safeguard)")

    if route == "general" and not tools_used and FORWARD_CLAIM_RE.search(draft or ""):
        trace("safeguard", "reply says the team was told, but nothing was logged: logging it now")
        record_feedback(user_message)
        tools_used.append("record_feedback (safeguard)")

    reply, reflection = reflect(user_message, route, draft, tools_used, tool_results, history)
    reply = enforce_language(reply, user_message, history)
    return {"reply": reply, "route": route, "tools_used": tools_used, "draft": draft, "reflection": reflection}


def agent_reply(message, history):
    """Entry point for chat front-ends (Gradio, Telegram): takes a message and history, returns the reply text."""
    try:
        return run_agent(message, history)["reply"]
    except QuotaBusy as busy:
        trace("quota", f"{busy}. {quota_status()}")
        return busy_message(message, normalize_history(history), busy.seconds)
    except Exception as exc:
        trace("error", f"{type(exc).__name__}: {exc}")
        return ("Sorry, I'm having a technical problem on my side right now. "
                "Please try again in a moment, or email hello@cedarsun-energy.example.")


def quick_flags(draft, route, tools_used):
    flags = []
    if re.search(r"(\$\s?\d|\d\s?(usd|dollars)\b|\bUSD\s?\d)", draft, re.I):
        flags.append("the draft mentions an amount of money; prices are not allowed")
    if re.search(r"\b(payback|pay back|you will save|savings of|save \d)", draft, re.I):
        flags.append("the draft talks about savings or payback")
    if re.search(r"\b\d+(\.\d+)?\s?(kwp|panels)\b", draft, re.I) and "estimate_system_size" not in tools_used:
        flags.append("sizing numbers appear but estimate_system_size was not called")
    if route == "lead" and "record_customer_interest" not in tools_used and "email" not in draft.lower():
        flags.append("lead conversation with no request for an email address")
    if len(draft.split()) > 170:
        flags.append("the reply is much longer than the 120-word guideline")
    return flags


REFLECTION_PROMPT = """\
You are a strict reviewer of a customer-service reply written by Zeina, CedarSun Energy's virtual advisor.
Check the DRAFT against the INSTRUCTIONS and the KNOWLEDGE BASE SUMMARY. Look for:
1. Invented facts or numbers (anything not in the knowledge base or the tool results).
2. Any price, discount, savings, payback or financing statement.
3. Sizing results that are not labelled approximate / subject to the free site survey, or that differ from the tool results.
4. Legal, regulatory, grid-export, tax or insurance claims, or electrical repair instructions.
5. A lead conversation that forgot to ask for a name or email.
6. A reply that is too long, has lists/headings, or does not end with one question or next step.
If the draft is fine, keep it unchanged. If not, rewrite it in Zeina's voice, fixing only what is wrong.
Keep the revised reply in the SAME language and script as the draft and never add numbers or facts that are not in the draft,
the tool results or the knowledge base. Only revise for real violations; style preferences are not violations.
Return ONLY a JSON object: {"verdict": "ok" or "revise", "issues": ["..."], "revised_reply": "the final reply text"}"""


def reflect(user_message, route, draft, tools_used, tool_results, history=None):
    """Check the draft against the instructions and revise it if needed. Returns (final_reply, record)."""
    if not REFLECTION_ENABLED or not draft:
        return draft, None
    flags = quick_flags(draft, route, tools_used)
    tool_text = "\n".join(f"{t['tool']}: {t['result'][:1500]}" for t in tool_results) or "(no tools used)"
    earlier = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in (history or [])[-6:]) or "(first message)"
    review_input = (f"INSTRUCTIONS:\n{INSTRUCTIONS}\n\nKNOWLEDGE BASE SUMMARY:\n{BUSINESS_SUMMARY}\n\n"
                    f"EARLIER CHAT (facts the customer already gave count as given):\n{earlier}\n\n"
                    f"CUSTOMER MESSAGE: {user_message}\nINTENT: {route}\nTOOLS USED: {tools_used}\n"
                    f"TOOL RESULTS:\n{tool_text}\nCODE-CHECK FLAGS: {flags or 'none'}\n\nDRAFT:\n{draft}")
    revised, issues = draft, []
    try:
        resp = chat(
            optional=True, temperature=0,
            messages=[{"role": "system", "content": REFLECTION_PROMPT}, {"role": "user", "content": review_input}])
        text = resp.choices[0].message.content or ""
        match = re.search(r"\{.*\}", text, re.S)
        data = json.loads(match.group(0)) if match else {}
        issues = data.get("issues") or []
        candidate = (data.get("revised_reply") or "").strip()
        if data.get("verdict") == "revise" and candidate:
            lang_before, lang_after = detect_language(draft), detect_language(candidate)
            if lang_before and lang_after and lang_before != lang_after:
                issues.append(f"revision rejected: it changed the language ({lang_before} -> {lang_after})")
            elif quick_flags(candidate, route, tools_used) and not quick_flags(draft, route, tools_used):
                issues.append("revision rejected: it introduced a rule violation")
            else:
                revised = candidate
    except Exception as exc:
        trace("reflection", f"skipped ({type(exc).__name__}: {exc}); sending the draft as is")
    record = {"user_message": user_message, "intent": route, "draft": draft, "revised": revised,
              "changed": revised != draft, "issues": issues, "code_flags": flags}
    _log("reflection.jsonl", record)
    if record["changed"]:
        trace("reflection", f"REVISED. Issues: {issues}\n   draft:   {draft}\n   revised: {revised}")
    else:
        trace("reflection", "draft approved unchanged")
    return revised, record


LOGO_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 120 120' width='52' height='52'>"
    "<defs><linearGradient id='sg' x1='0' y1='0' x2='1' y2='1'><stop offset='0' stop-color='#FFC247'/><stop offset='1' stop-color='#F08A1C'/></linearGradient>"
    "<linearGradient id='tg' x1='0' y1='0' x2='0' y2='1'><stop offset='0' stop-color='#2A6BB5'/><stop offset='1' stop-color='#173F73'/></linearGradient></defs>"
    "<g stroke='url(#sg)' stroke-width='5' stroke-linecap='round'><line x1='74' y1='8' x2='74' y2='16'/><line x1='104' y1='20' x2='98' y2='26'/>"
    "<line x1='116' y1='48' x2='108' y2='48'/><line x1='104' y1='76' x2='98' y2='70'/><line x1='48' y1='20' x2='54' y2='26'/><line x1='74' y1='88' x2='74' y2='80'/></g>"
    "<circle cx='74' cy='48' r='23' fill='url(#sg)'/>"
    "<g fill='url(#tg)'><rect x='35' y='86' width='8' height='26' rx='2'/><ellipse cx='39' cy='92' rx='31' ry='7.5'/><ellipse cx='39' cy='78' rx='27' ry='7.5'/>"
    "<ellipse cx='39' cy='64' rx='22' ry='7'/><ellipse cx='39' cy='51' rx='17' ry='6.5'/><ellipse cx='39' cy='39' rx='12' ry='6'/>"
    "<ellipse cx='39' cy='28' rx='7' ry='5.5'/><path d='M36 14 Q39 6 42 14 Z'/></g></svg>"
)
CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Sans+Arabic:wght@400;500;600&display=swap');

:root, .gradio-container {
  --cs-ink:#0E1A2B; --cs-night:#12294A; --cs-night-2:#1B3A66; --cs-sun:#F2A33A; --cs-sun-deep:#B86A0B;
  --cs-paper:#F4F6F9; --cs-white:#FFFFFF; --cs-line:#D8DFE8; --cs-muted:#51607A; --cs-live:#1C9B63;
  color-scheme: light;
}
html, body, .gradio-container, gradio-app {background: var(--cs-paper) !important;}
.gradio-container {
  max-width: 980px !important; width: 100% !important; margin: auto; color: var(--cs-ink);
  font-family: 'IBM Plex Sans', 'IBM Plex Sans Arabic', ui-sans-serif, system-ui, 'Segoe UI', Tahoma, sans-serif !important;
}
.gradio-container *:not(svg *):not(code) {font-family: inherit;}
.gradio-container .block, .gradio-container .form {border: 0 !important; box-shadow: none !important; background: transparent !important;}
.gradio-container .gap {gap: 0 !important;}
:focus-visible {outline: 3px solid var(--cs-sun) !important; outline-offset: 2px;}
footer {display: none !important;}

/* top bar */
#brand {display:flex; align-items:center; gap:12px; padding:18px 4px 16px;}
#brand svg {width:40px; height:40px; flex:none;}
#brand .name {font-weight:600; font-size:1.08rem; letter-spacing:-.01em; color: var(--cs-ink);}
#brand .live {margin-left:auto; display:flex; align-items:center; gap:8px; font-size:.9rem; color: var(--cs-muted);}
.dot {display:inline-block; width:9px; height:9px; border-radius:50%; background: var(--cs-live); box-shadow: 0 0 0 3px rgba(28,155,99,.18);}

/* hero */
#hero {background: var(--cs-night); border-radius: 10px; padding: 40px 40px 36px !important; gap: 0 !important; color:#fff;}
#hero .block {background: transparent !important;}
.hero-grid {display:grid; grid-template-columns: 1.15fr .85fr; gap: 40px; align-items:center; margin-bottom: 28px;}
.hero-grid h1 {margin:0 0 14px; font-size: clamp(1.85rem, 4.6vw, 2.75rem); line-height:1.08; font-weight:600; letter-spacing:-.025em; color:#fff !important; max-width: 17ch;}
.hero-grid p {margin:0; font-size:1.06rem; line-height:1.55; color:#C9D6EA !important; max-width: 46ch;}
.sunpath {margin:0;}
.sunpath svg {width:100%; height:auto; display:block;}
.sunpath figcaption {margin-top:8px; font-size:.9rem; line-height:1.45; color:#C9D6EA;}
.sunpath .horizon {stroke:#7F95B8; stroke-width:1;}
.sunpath .full {stroke:#7F95B8; stroke-width:1.5; fill:none; stroke-dasharray:2 5;}
.sunpath .strong {stroke: var(--cs-sun); stroke-width:5; fill:none; stroke-linecap:round;
                  stroke-dasharray: 42 58; stroke-dashoffset: -29; animation: draw 2.2s cubic-bezier(.3,.7,.2,1) .25s both;}
.sunpath text {fill:#9FB4D4; font-size:11px;}
@keyframes draw {from {stroke-dasharray: 0 100;} to {stroke-dasharray: 42 58;}}
@media (prefers-reduced-motion: reduce) {.sunpath .strong {animation:none;}}

#callbtn, #callbtn button {background: var(--cs-sun) !important; color: var(--cs-ink) !important; border: 0 !important; font-weight:600 !important;
        font-size:1.05rem !important; padding: 15px 26px !important; border-radius: 8px !important; width:auto !important; min-width: 270px; align-self:flex-start;}
#callbtn:hover, #callbtn button:hover {background:#F7B454 !important;}

/* facts strip: one ruled row, not four cards */
.facts {display:grid; grid-template-columns: repeat(4, 1fr); margin: 26px 0 8px; border-top:1px solid var(--cs-line); border-bottom:1px solid var(--cs-line);}
.facts div {padding:16px 18px 16px 0; margin-right:18px; border-right:1px solid var(--cs-line);}
.facts div:last-child {border-right:0; margin-right:0;}
.facts b {display:block; font-size:1.12rem; font-weight:600; letter-spacing:-.01em;}
.facts span {display:block; margin-top:3px; font-size:.88rem; color: var(--cs-muted); line-height:1.4;}

h2.section {margin: 26px 0 4px; font-size:1.2rem; font-weight:600; letter-spacing:-.015em; color: var(--cs-ink) !important;}
p.sub {margin:0 0 6px; color: var(--cs-muted) !important; font-size:.95rem;}

/* question list */
#qlist {gap:0 !important; margin-bottom: 6px;}
#qlist button {display:flex !important; align-items:center; justify-content:space-between; width:100% !important; text-align:left; background:transparent !important;
        color: var(--cs-ink) !important; border:0 !important; border-bottom:1px solid var(--cs-line) !important; border-radius:0 !important;
        padding: 16px 4px !important; font-size:1.02rem !important; font-weight:500 !important;}
#qlist button::after {content:''; flex:none; width:8px; height:8px; margin-left:16px; border-right:2px solid var(--cs-sun-deep); border-top:2px solid var(--cs-sun-deep); transform: rotate(45deg);}
#qlist button:hover {background: #E9EEF5 !important;}

/* what happens next: a real sequence */
.steps {list-style:none; margin:14px 0 6px; padding:0; display:grid; grid-template-columns: repeat(4, 1fr); gap: 0; position:relative;}
.steps::before {content:''; position:absolute; left:14px; right:14px; top:14px; height:1px; background: var(--cs-line);}
.steps li {position:relative; padding: 0 16px 0 0;}
.steps .n {position:relative; display:flex; align-items:center; justify-content:center; width:29px; height:29px; border-radius:50%; background: var(--cs-paper);
           border:1.5px solid var(--cs-night); font-size:.82rem; font-weight:600; color: var(--cs-night); margin-bottom:10px;}
.steps b {display:block; font-size:.98rem; font-weight:600;}
.steps span {display:block; margin-top:2px; font-size:.88rem; color: var(--cs-muted); line-height:1.4;}
.fineprint {margin: 30px 0 18px; font-size:.84rem; color: var(--cs-muted); text-align:left;}

/* ringing call screen */
.call-screen {text-align:center; padding:44px 20px 40px; margin:18px auto; max-width:460px; border-radius:12px; background: var(--cs-white); border:1px solid var(--cs-line);}
.call-screen h2 {margin:24px 0 4px; font-size:1.35rem; font-weight:600; color: var(--cs-ink) !important;}
.call-screen p {margin:0; color: var(--cs-muted) !important;}
.ring {position:relative; width:96px; height:96px; margin:auto;}
.ring .phone {position:relative; z-index:2; width:96px; height:96px; border-radius:50%; background: var(--cs-night);
              display:flex; align-items:center; justify-content:center; animation:shake 1.2s ease-in-out infinite;}
.ring .phone svg {width:42px; height:42px;}
.ring span {position:absolute; inset:0; border-radius:50%; border:3px solid var(--cs-sun); opacity:0; animation:pulse 1.8s ease-out infinite;}
.ring span:nth-child(2) {animation-delay:.6s;} .ring span:nth-child(3) {animation-delay:1.2s;}
.connected .ring span {animation:none; opacity:0;}
.connected .ring .phone {animation:none; background: var(--cs-live);}
.dots::after {content:''; animation:dots 1.4s steps(4,end) infinite;}
@keyframes dots {0% {content:'';} 25% {content:'.';} 50% {content:'..';} 75% {content:'...';}}
@keyframes shake {0%,100% {transform:rotate(0);} 10%,30% {transform:rotate(-12deg);} 20%,40% {transform:rotate(12deg);} 50% {transform:rotate(0);}}
@keyframes pulse {0% {transform:scale(.9); opacity:.8;} 100% {transform:scale(1.9); opacity:0;}}
@media (prefers-reduced-motion: reduce) {.ring .phone, .ring span, .dots::after {animation:none;}}

/* conversation */
#chatwrap {background: var(--cs-white); border:1px solid var(--cs-line); border-radius:12px; overflow:hidden; gap:0 !important;}
#chatbar {display:flex; align-items:center; gap:12px; padding:12px 16px; border-bottom:1px solid var(--cs-line); background: var(--cs-white);}
#chatbar .who {display:flex; align-items:center; gap:12px; flex:1;}
#chatbar .av {width:40px; height:40px; border-radius:50%; background: var(--cs-night); display:flex; align-items:center; justify-content:center; flex:none;}
#chatbar .av svg {width:30px; height:30px; background:#fff; border-radius:50%; padding:2px;}
#chatbar b {display:block; font-weight:600; font-size:1rem;}
#chatbar small {display:flex; align-items:center; gap:6px; color: var(--cs-muted); font-size:.85rem;}
#endbtn {flex:none; width:auto !important; min-width:0 !important; background:transparent !important; color: var(--cs-muted) !important;
         border:1px solid var(--cs-line) !important; border-radius:8px !important; font-size:.9rem !important; padding: 8px 14px !important; font-weight:500 !important;}
#endbtn:hover {color: var(--cs-ink) !important; border-color: var(--cs-muted) !important;}
#chatrow {gap:0 !important; align-items:center; border-bottom:1px solid var(--cs-line);}
#zchat {border:0 !important; background: var(--cs-white) !important;}
#zchat .message {font-size:15px !important; line-height:1.5 !important; box-shadow:none !important; border-radius:14px !important;}
#zchat .user .message, #zchat .message.user {background: var(--cs-night) !important; color:#fff !important; border:0 !important;}
#zchat .user .message *, #zchat .message.user * {color:#fff !important;}
#zchat .bot .message, #zchat .message.bot {background: #EEF2F7 !important; color: var(--cs-ink) !important; border:0 !important;}
#zchat .placeholder-content, #zchat .placeholder {color: var(--cs-muted) !important;}
.gradio-container textarea, .gradio-container input[type=text] {font-size:15px !important; border:1px solid var(--cs-line) !important; border-radius:10px !important; background:#fff !important; color: var(--cs-ink) !important;}
.gradio-container textarea::placeholder {color:#64748b !important; opacity:1 !important;}
.gradio-container button.submit-button, .gradio-container .submit-button {background: var(--cs-sun) !important; color: var(--cs-ink) !important; border-radius:10px !important;}
.embedded .gradio-container {padding-top:6px !important;}

/* polish pass */
#callbtn {margin-top: 30px !important;}
#qlist button {padding-left: 0 !important; padding-right: 0 !important;}
#qlist button:hover {padding-left: 8px !important;}
#chatwrap {width:100% !important; min-width:100% !important;}
#chatwrap > .form, #chatwrap form {padding: 12px 14px 14px !important; background:#fff !important; border-top:1px solid var(--cs-line) !important;}
#chatwrap textarea {padding: 12px 14px !important; min-height: 46px;}
#chatwrap textarea:focus {border-color: var(--cs-sun-deep) !important; box-shadow: 0 0 0 3px rgba(242,163,58,.25) !important;}
.gradio-container .html-container {padding: 0 !important;}
#chatbar {border-bottom: 0 !important;}
#chatrow {margin: 0 !important;}
.gradio-container .block.padded, .gradio-container .prose {padding: 0 !important;}
#hero .prose, #hero .block {padding: 0 !important;}
#qlist button, #qlist .block, #callbtn, #endbtn {box-shadow: none !important;}
#qlist button {transition: background .15s;}
#qlist button:hover {padding-left: 10px !important;}
#brand {border-bottom: 1px solid var(--cs-line); margin-bottom: 16px;}
#brand .name, #brand .live {white-space: nowrap;}
#chatrow {display:flex !important; flex-direction:row !important; align-items:center; flex-wrap:nowrap !important;
          padding-right: 14px; border-bottom:1px solid var(--cs-line); background:#fff !important;}
#chatrow > * {min-width:0;}
#chatrow > :first-child {flex:1 1 auto !important;}
#chatrow #endbtn {flex:none !important;}
#zchat .user .message-content, #zchat .user .message-content * , #zchat .user p {color:#fff !important;}
#zchat .bot .message-content, #zchat .bot .message-content *, #zchat .bot p {color: var(--cs-ink) !important;}
#zchat .message {padding: 10px 14px !important;}

@media (max-width: 760px) {
  #hero {padding: 26px 20px 24px !important;}
  .hero-grid {grid-template-columns: 1fr; gap: 22px;}
  .facts {grid-template-columns: 1fr 1fr;}
  .facts div:nth-child(2) {border-right:0; margin-right:0;}
  .facts div:nth-child(-n+2) {border-bottom:1px solid var(--cs-line);}
  .steps {grid-template-columns: 1fr; gap: 18px;}
  .steps::before {left:14px; right:auto; top:14px; bottom:14px; width:1px; height:auto;}
  .steps li {display:grid; grid-template-columns: 29px 1fr; column-gap:14px; padding:0;}
  .steps .n {margin-bottom:0; grid-row: span 2;}
  #callbtn, #callbtn button {width:100% !important; min-width:0;}
  #brand .live {font-size:.8rem;}
  #brand .name {font-size:1rem;}
  .gradio-container {padding-left:12px !important; padding-right:12px !important;}
}
"""
# Compact layout used only inside the website's chat window (an iframe about 400 x 560 px).
EMBED_CSS = """
html, body {overflow-x: hidden !important; overflow-y: auto !important;}   /* the landing page scrolls; the chat screen fits the window */
.gradio-container {padding: 4px 8px !important; min-height: 0 !important; max-width: 100% !important;}
footer {display: none !important;}
.progress-text, .meta-text, .meta-text-center, .eta-bar {display: none !important;}
#chatwrap {border-radius: 10px;}
#zchat, #zchat .wrapper, #zchat .bubble-wrap {height: calc(100vh - 290px) !important; max-height: calc(100vh - 290px) !important;}
#zchat .message-row {max-width: 100% !important; overflow: hidden !important;}
#zchat .message, #zchat .message * {font-size: 14px !important; line-height: 1.4 !important;}
#zchat .message {max-width: 100% !important; width: auto !important; overflow-wrap: break-word !important; box-sizing: border-box !important;}
#zchat .flex-wrap {min-width: 0 !important;}
textarea, input[type=text] {font-size: 14px !important;}
.call-screen {padding: 20px 12px !important;}
"""

HEADER = (f"<div id='brand'>{LOGO_SVG}<span class='name'>CedarSun Energy</span>"
          "<span class='live'><span class='dot'></span>Advisors online</span></div>")

SUN_PATH_SVG = (
    "<svg viewBox='0 0 360 200' role='img' aria-label='The sun crosses the sky from sunrise to sunset; the strongest five hours are highlighted'>"
    "<line class='horizon' x1='10' y1='170' x2='350' y2='170'/>"
    "<path class='full' pathLength='100' d='M20 170 Q180 -70 340 170'/>"
    "<path class='strong' pathLength='100' d='M20 170 Q180 -70 340 170'/>"
    "<circle cx='180' cy='50' r='22' fill='#F2A33A' opacity='.22'/><circle cx='180' cy='50' r='12' fill='#F2A33A'/>"
    "<text x='20' y='190'>Sunrise</text><text x='340' y='190' text-anchor='end'>Sunset</text></svg>")

HERO_HTML = (
    "<div class='hero-grid'><div>"
    "<h1>Tell us what runs in your home. We'll size the solar.</h1>"
    "<p>Chat with Zeina, CedarSun's solar advisor, for a rough system size in a few minutes. "
    "A free site survey comes next, with no obligation. Ask in English, French or Arabic.</p></div>"
    f"<figure class='sunpath'>{SUN_PATH_SVG}"
    "<figcaption>Beirut gets about 5 hours of strong sun a day. We size your system around that.</figcaption></figure></div>")

FACTS_HTML = ("<div class='facts'>"
              "<div><b>380+ homes</b><span>powered since 2019</span></div>"
              "<div><b>Free site survey</b><span>no obligation</span></div>"
              "<div><b>10-year warranty</b><span>on batteries and inverters</span></div>"
              "<div><b>A call within 1 business day</b><span>from a specialist</span></div></div>")

STEPS_HTML = ("<h2 class='section'>What happens after you chat</h2>"
              "<ol class='steps'>"
              "<li><span class='n'>1</span><b>A specialist calls you</b><span>Within 1 business day</span></li>"
              "<li><span class='n'>2</span><b>Free site survey</b><span>Usually within 5 business days</span></li>"
              "<li><span class='n'>3</span><b>Written quotation</b><span>Within 3 business days of the survey</span></li>"
              "<li><span class='n'>4</span><b>Installation</b><span>Typically 1 to 3 days once approved</span></li></ol>")

CHATBAR_HTML = (f"<div id='chatbar'><div class='who'><span class='av'>{LOGO_SVG}</span>"
                "<div><b>Zeina</b><small><span class='dot'></span>CedarSun solar advisor, online</small></div></div></div>")

SUGGESTED_QUESTIONS = ["How many solar panels do I need for my apartment?",
                       "What warranty do you give on batteries?",
                       "How does the free site survey work?",
                       "Can I sell extra power back to the grid?"]
RING_SECONDS, PICKUP_SECONDS = 4.0, 2.5   # how long the "call" rings before Zeina picks up


PHONE_SVG = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'>"
             "<path d='M6.6 10.8a15.1 15.1 0 0 0 6.6 6.6l2.2-2.2a1 1 0 0 1 1-.25 11.4 11.4 0 0 0 3.6.57 1 1 0 0 1 1 1V20a1 1 0 0 1-1 1"
             "A17 17 0 0 1 3 4a1 1 0 0 1 1-1h3.5a1 1 0 0 1 1 1c0 1.25.2 2.45.57 3.6a1 1 0 0 1-.25 1z'/></svg>")


def _call_html(title, subtitle, connected=False):
    spans = "" if connected else "<span></span><span></span><span></span>"
    dots = "" if connected else " dots"
    return (f"<div class='call-screen {'connected' if connected else ''}'><div class='ring'>{spans}"
            f"<div class='phone'>{PHONE_SVG}</div></div><h2 class='{dots.strip()}'>{title}</h2><p>{subtitle}</p></div>")


def submit_lead(name: str, email: str, area: str = "", message: str = "") -> str:
    """Used by the website's contact form: saves the lead exactly like the chat tool does."""
    note = f"[Website form] Area: {area or 'not given'}. {message or ''}".strip()
    return json.dumps(record_customer_interest(email=email, name=name, message=note))


def build_demo():
    """Landing page -> ringing call screen -> the Gradio ChatInterface with Zeina."""
    import gradio as gr
    avatar = BASE_DIR / "logo.svg"
    if not avatar.exists():
        avatar.write_text(LOGO_SVG, encoding="utf-8")

    def show(landing_on, call_on, chat_on):
        return gr.update(visible=landing_on), gr.update(visible=call_on), gr.update(visible=chat_on)

    def connect(question=None):
        """Return a generator: ringing screen first, then the chat (already asking `question` if one was picked)."""
        def run():
            yield (*show(False, True, False), _call_html("Calling a solar advisor", "Please hold, we are finding someone available"), [], [])
            time.sleep(RING_SECONDS)
            yield (*show(False, True, False), _call_html("Advisor found", "Connecting you now", connected=True), [], [])
            time.sleep(PICKUP_SECONDS)
            yield (*show(False, False, True), "", [], [])
            if question:
                asked = [{"role": "user", "content": question}]
                yield (*show(False, False, True), "", asked + [{"role": "assistant", "content": "typing..."}], asked)
                done = asked + [{"role": "assistant", "content": agent_reply(question, [])}]
                yield (*show(False, False, True), "", done, done)
        return run

    def end_call():
        return (*show(True, False, False), "", [], [])

    # The page is always light so the brand colours stay as designed. Inside the website's chat window the site already
    # shows the brand header, so hide ours (retry while the page renders).
    embed_js = ("() => { const kill = () => document.body.classList.remove('dark'); kill(); "
                "new MutationObserver(kill).observe(document.body, {attributes: true, attributeFilter: ['class']}); "
                "if (window.self === window.top) return; const css = " + json.dumps(EMBED_CSS) + "; "
                "const s = document.createElement('style'); s.textContent = css; document.head.appendChild(s); "
                "let n = 0; const t = setInterval(() => { const b = document.getElementById('brand'); if (b) b.style.display = 'none'; "
                "if (b || ++n > 40) clearInterval(t); }, 150); }")
    with gr.Blocks(title="CedarSun Energy | Customer Support", css=CSS, js=embed_js,
                   theme=gr.themes.Soft(primary_hue="amber", neutral_hue="slate")) as demo:
        gr.HTML(HEADER)

        with gr.Column(visible=True) as landing:
            with gr.Column(elem_id="hero"):
                gr.HTML(HERO_HTML)
                talk = gr.Button("Talk to a solar advisor", variant="primary", elem_id="callbtn")
            gr.HTML(FACTS_HTML)
            gr.HTML("<h2 class='section'>Start with a question</h2><p class='sub'>Pick one and Zeina answers right away.</p>")
            with gr.Column(elem_id="qlist"):
                question_buttons = [gr.Button(q) for q in SUGGESTED_QUESTIONS]
            gr.HTML(STEPS_HTML)

        with gr.Column(visible=False) as calling:
            call_view = gr.HTML()

        with gr.Column(visible=False, elem_id="chatwrap") as chat_col:
            with gr.Row(elem_id="chatrow"):
                gr.HTML(CHATBAR_HTML)
                hang_up = gr.Button("End conversation", size="sm", elem_id="endbtn")
            chat = gr.ChatInterface(
                fn=agent_reply,
                type="messages",
                chatbot=gr.Chatbot(type="messages", height=460, show_label=False, elem_id="zchat",
                                   placeholder="<b>Zeina</b> from CedarSun Energy<br>Send a message to start chatting.",
                                   avatar_images=(None, str(avatar))),
                textbox=gr.Textbox(placeholder="Type your message to Zeina...", container=False, scale=7, lines=1, max_lines=4),
            )

        gr.HTML("<p class='fineprint'>CedarSun Energy is a fictional company (university project). Beirut, Lebanon.</p>")

        # The ChatInterface keeps its own copy of the history, so both the visible chatbot and that state are updated together.
        outputs = [landing, calling, chat_col, call_view, chat.chatbot, chat.chatbot_state]
        talk.click(connect(), None, outputs)
        for btn, q in zip(question_buttons, SUGGESTED_QUESTIONS):
            btn.click(connect(q), None, outputs)
        hang_up.click(end_call, None, outputs)
        gr.api(submit_lead, api_name="lead")      # the website's contact form posts here
    return demo


if __name__ == "__main__":
    build_demo().launch()
