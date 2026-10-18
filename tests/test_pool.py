"""Offline tests for the key pool: no API key, no internet, no waiting (a fake clock replaces time).
Run:  python tests/test_pool.py
"""
import os, sys, tempfile, json
from pathlib import Path
from types import SimpleNamespace

os.environ["LOG_DIR"] = tempfile.mkdtemp()
os.environ["VERBOSE"] = "0"
# Blank (not remove) every key and pool setting: python-dotenv fills in anything that is merely missing from a local .env,
# which would give the pool extra keys and break the one-key expectations below. An empty value stays empty.
for name in ("GEMINI_API_KEY", "GEMINI_API_KEYS", "OPENAI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "CEREBRAS_API_KEY", "LLM_FALLBACKS",
             *[f"GEMINI_API_KEY_{n}" for n in range(2, 10)], *[f"GEMINI_API_KEY{n}" for n in range(2, 10)]):
    os.environ[name] = ""
os.environ["RPM_LIMIT"] = "0"
os.environ["LLM_PROVIDER"] = "gemini"
os.environ["LLM_MODEL"] = "gemini-test"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # noqa: E402

GEMINI_429 = ("Error code: 429 - [{'error': {'code': 429, 'message': 'You exceeded your current quota. Quota exceeded for metric: "
              "generate_content_free_tier_requests, limit: 5, model: gemini-test\\nPlease retry in %s.', 'status': 'RESOURCE_EXHAUSTED', "
              "'details': [{'quotaId': '%s'}, {'@type': 'RetryInfo', 'retryDelay': '%s'}]}}]")


class Err(Exception):
    def __init__(self, text, status_code=None):
        super().__init__(text)
        self.status_code = status_code


def minute_429(delay="34.5s"):
    return Err(GEMINI_429 % (delay, "GenerateRequestsPerMinutePerProjectPerModel-FreeTier", delay.split(".")[0] + "s"), 429)


def daily_429():
    return Err(GEMINI_429 % ("3h", "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "10800s"), 429)


class Clock:
    def __init__(self): self.t, self.slept = 1000.0, []
    def now(self): return self.t
    def sleep(self, s): self.slept.append(s); self.t += s


class FakeClients:
    """One fake client per key. `script[key]` is a list of outcomes: an Exception to raise, or a string to return."""
    def __init__(self, script):
        self.script, self.calls = script, []

    def client_for(self, ep, single):
        def create(model, **kw):
            self.calls.append((ep.key, model))
            outcome = self.script[ep.key].pop(0) if len(self.script[ep.key]) > 1 else self.script[ep.key][0]
            if isinstance(outcome, Exception):
                raise outcome
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=outcome, tool_calls=None))])
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def setup(keys, script, rpm=0, fallbacks=""):
    """Reset all pool state, set the keys and install the fakes."""
    os.environ["GEMINI_API_KEYS"] = ",".join(keys)
    app.RPM_LIMIT, app.QUOTA_MAX_WAIT = rpm, 8.0
    app.FALLBACKS = app._parse_fallbacks(fallbacks)
    app._cooldown.clear(); app._cooldown_kind.clear(); app._calls.clear(); app._clients.clear()
    clock, fake = Clock(), FakeClients(script)
    app._clock, app._sleep, app._client_for = clock.now, clock.sleep, fake.client_for
    return clock, fake


def reply(resp): return resp.choices[0].message.content


def test_keys_are_read_from_several_variables():
    app.FALLBACKS = []
    os.environ["GEMINI_API_KEYS"] = "k1, k2 ,k1,k3"
    os.environ["GEMINI_API_KEY"] = "k0"
    assert app.provider_keys(app.PROVIDERS["gemini"]) == ["k0", "k1", "k2", "k3"]
    del os.environ["GEMINI_API_KEY"]
    labels = [e.label for e in app._endpoints()]
    assert len(labels) == 3 and labels[0] == "gemini#1:gemini-test", labels
    assert not any("k1" in l for l in labels)          # labels never contain a key


def test_retry_delay_is_parsed():
    assert abs(app._retry_seconds(Err("Please retry in 49.794986ms.")) - 0.0498) < 0.001
    assert app._retry_seconds(Err("Please retry in 12.5s")) == 12.5
    assert app._retry_seconds(Err("{'retryDelay': '34s'}")) == 34
    assert app._retry_seconds(Err("nothing here")) == 0


def test_second_key_takes_over_and_the_first_one_rests():
    clock, fake = setup(["k1", "k2"], {"k1": [minute_429("34.5s")], "k2": ["hello from k2"]})
    assert reply(app.chat(messages=[])) == "hello from k2"
    assert [c[0] for c in fake.calls] == ["k1", "k2"]
    assert 34 <= app._cooldown[("gemini", "gemini-test", 0)] - clock.t <= 36
    fake.calls.clear()
    assert reply(app.chat(messages=[])) == "hello from k2"
    assert [c[0] for c in fake.calls] == ["k2"], "a resting key must not be called again"
    clock.t += 40                                         # its window has reopened
    fake.script["k1"] = ["hello from k1"]
    fake.calls.clear()
    assert reply(app.chat(messages=[])) == "hello from k1" and fake.calls[0][0] == "k1"


def test_tiny_retry_delay_still_rests_for_the_minimum():
    clock, fake = setup(["k1", "k2"], {"k1": [Err(GEMINI_429 % ("49.79ms", "PerMinute", "0s"), 429)], "k2": ["ok"]})
    app.chat(messages=[])
    assert app._cooldown[("gemini", "gemini-test", 0)] - clock.t >= app.MIN_QUOTA_COOLDOWN - 0.1


def test_all_keys_busy_raises_with_the_soonest_wait():
    clock, fake = setup(["k1", "k2"], {"k1": [minute_429("40s")], "k2": [minute_429("20s")]})
    try:
        app.chat(messages=[]); assert False, "expected QuotaBusy"
    except app.QuotaBusy as busy:
        assert 19 <= busy.seconds <= 21, busy.seconds
    assert clock.slept == [], "must not wait 20s inside a reply"


def test_short_wait_is_waited_out_once():
    clock, fake = setup(["k1"], {"k1": [minute_429("1s"), "ok after the wait"]})
    app.MIN_QUOTA_COOLDOWN = 3                              # pretend the provider asked for a short pause
    try:
        assert reply(app.chat(messages=[])) == "ok after the wait"
        assert clock.slept and 2.5 <= clock.slept[0] <= 4, clock.slept
    finally:
        app.MIN_QUOTA_COOLDOWN = 15


def test_optional_calls_never_wait():
    clock, fake = setup(["k1"], {"k1": [minute_429("3s"), "unused"]})
    app.MIN_QUOTA_COOLDOWN = 3
    app.QUOTA_MAX_WAIT = 0.0                                # the main call may not wait either, so the key is left resting
    try:
        app.chat(messages=[])
    except app.QuotaBusy:
        pass
    app.QUOTA_MAX_WAIT = 8.0                                # a main call could now wait 3s, but an optional one must not
    try:
        app.chat(optional=True, messages=[]); assert False
    except app.QuotaBusy:
        pass
    finally:
        app.MIN_QUOTA_COOLDOWN = 15
    assert clock.slept == []


def test_daily_limit_rests_for_hours_and_says_so():
    clock, fake = setup(["k1"], {"k1": [daily_429()]})
    try:
        app.chat(messages=[]); assert False
    except app.QuotaBusy as busy:
        assert busy.seconds >= 3600
    text = app.busy_message("hello there how are you", [], 3 * 3600)
    assert "a few hours" in text


def test_bad_key_rotates_and_rests_for_an_hour():
    clock, fake = setup(["bad", "good"], {"bad": [Err("API key not valid", 403)], "good": ["fine"]})
    assert reply(app.chat(messages=[])) == "fine"
    assert app._cooldown_kind[("gemini", "gemini-test", 0)] == "auth"
    assert app._cooldown[("gemini", "gemini-test", 0)] - clock.t >= 3500


def test_only_a_bad_key_is_a_real_error_not_a_busy_message():
    clock, fake = setup(["bad"], {"bad": [Err("API key not valid", 403)]})
    try:
        app.chat(messages=[]); assert False
    except app.QuotaBusy:
        assert False, "a rejected key is not a rate limit"
    except Err:
        pass


def test_request_errors_do_not_rest_the_key():
    clock, fake = setup(["k1"], {"k1": [Err("tool_choice is not supported", 400), "second try works"]})
    try:
        app.chat(messages=[])
    except Err:
        pass
    assert app._cooldown == {}
    assert reply(app.chat(messages=[])) == "second try works"


def test_proactive_rpm_limit_avoids_the_429():
    clock, fake = setup(["k1"], {"k1": ["a", "b", "c"]}, rpm=2)
    app.chat(messages=[]); app.chat(messages=[])
    assert len(fake.calls) == 2
    try:
        app.chat(messages=[]); assert False
    except app.QuotaBusy as busy:
        assert 55 <= busy.seconds <= 61
    assert len(fake.calls) == 2, "the third request must be held back, not sent"
    clock.t += 61
    assert reply(app.chat(messages=[])) == "c" and len(fake.calls) == 3


def test_rpm_limit_keeps_room_for_the_main_reply():
    clock, fake = setup(["k1"], {"k1": ["x"] * 10}, rpm=5)
    for _ in range(3):
        app.chat(optional=True, messages=[])                # helpers may use 3 of 5
    try:
        app.chat(optional=True, messages=[]); assert False
    except app.QuotaBusy:
        pass
    assert reply(app.chat(messages=[])) == "x"              # the main call still fits


def test_fallback_models_get_their_own_quota():
    clock, fake = setup(["k1"], {"k1": [minute_429("30s"), "answered by the fallback model"]}, fallbacks="gemini:gemini-lite")
    assert reply(app.chat(messages=[])) == "answered by the fallback model"
    assert [c[1] for c in fake.calls] == ["gemini-test", "gemini-lite"]


def test_customer_sees_a_friendly_message_in_their_language():
    clock, fake = setup(["k1"], {"k1": [minute_429("40s")]})
    cases = {
        "How many panels do I need for my flat in Beirut?": "send your message again in about",
        "Bonjour, je voudrais savoir combien de panneaux il faut pour mon appartement": "renvoyer votre message",
        "كم لوح بدي لبيتي؟ عندي براد وتكييف": "ابعتلي رسالتك",
        "kifak, bade ktir ma3lomet 3an el panneaux": "Ib3atli resaltak",
    }
    for message, expected in cases.items():
        out = app.agent_reply(message, [])
        assert expected in out, (message, out)
        assert "technical problem" not in out
        assert "hello@cedarsun-energy.example" in out
        clock.t += 100                                       # let the key reopen so each case starts fresh
        fake.script["k1"] = [minute_429("40s")]
        app._cooldown.clear(); app._cooldown_kind.clear()


def test_wait_wording():
    assert "40 seconds" in app.busy_message("hello there how are you", [], 38)
    assert "3 minutes" in app.busy_message("hello there how are you", [], 150)
    assert "40 sanye" in app.busy_message("kifak shu badak", [], 38)


def test_other_failures_keep_the_generic_message():
    clock, fake = setup(["k1"], {"k1": [Err("boom", 400)]})
    out = app.agent_reply("How many panels do I need for my flat in Beirut?", [])
    assert "technical problem" in out


def test_same_key_listed_under_two_provider_names_counts_once():
    """gemini2 / gemini3 are old aliases for GEMINI_API_KEY_2 / _3, which the pool already picks up under 'gemini'."""
    saved = os.environ.get("GEMINI_API_KEYS", "")
    os.environ["GEMINI_API_KEYS"] = ""                       # earlier tests may have left a plural list behind
    os.environ["GEMINI_API_KEY"], os.environ["GEMINI_API_KEY_2"], os.environ["GEMINI_API_KEY2"] = "k-one", "k-two", "k-two"
    try:
        app.FALLBACKS = app._parse_fallbacks("gemini2:gemini-test")
        labels = [(e.provider, e.key) for e in app._endpoints() if e.model == "gemini-test"]
        assert len(labels) == len({key for _, key in labels}) == 2, labels      # k-one and k-two, once each
    finally:
        os.environ["GEMINI_API_KEY"] = os.environ["GEMINI_API_KEY_2"] = os.environ["GEMINI_API_KEY2"] = ""
        os.environ["GEMINI_API_KEYS"] = saved
        app.FALLBACKS = []


def test_short_retry_delay_is_a_minute_limit_even_if_the_text_says_per_day():
    """Gemini error texts mention the per-day quota even when only the per-minute one was hit: trust the retry delay."""
    exc = Err("Quota exceeded for metric: generate_content_free_tier_requests, per day limit: 20. Please retry in 30s.", 429)
    assert app._cooldown_seconds("quota", exc) == 30
    assert app._cooldown_seconds("quota", Err("per day limit. Please retry in 2s.", 429)) == app.MIN_QUOTA_COOLDOWN


def test_daily_limit_without_a_delay_is_rechecked_after_a_while_not_parked_for_hours():
    exc = Err("Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier", 429)
    assert app._cooldown_seconds("quota", exc) == app.DAILY_RECHECK < app.DAILY_COOLDOWN
    assert app._cooldown_seconds("quota", daily_429()) == 10800          # an explicit long delay is respected, capped at DAILY_COOLDOWN


def test_logs_endpoint_is_closed_without_the_secret():
    saved = os.environ.get("LOGS_TOKEN", "")
    try:
        os.environ["LOGS_TOKEN"] = ""
        assert json.loads(app.read_logs("")) == {"error": "forbidden"}                 # no secret configured: switched off
        assert json.loads(app.read_logs("anything")) == {"error": "forbidden"}
        os.environ["LOGS_TOKEN"] = "s3cret"
        assert json.loads(app.read_logs("wrong")) == {"error": "forbidden"}
        app.record_feedback("log endpoint test question")
        data = json.loads(app.read_logs("s3cret"))
        assert "feedback.jsonl" in data and "log endpoint test question" in data["feedback.jsonl"]
    finally:
        os.environ["LOGS_TOKEN"] = saved


def test_log_sync_does_nothing_unless_configured():
    started = []
    import threading
    real = threading.Thread
    threading.Thread = lambda *a, **k: started.append(1) or real(target=lambda: None)
    try:
        app.LOGS_REPO = ""
        app._sync_log("feedback.jsonl")
        assert started == []                                    # no LOGS_REPO: no background upload is started
    finally:
        threading.Thread = real


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("PASS", name)
    print(f"\nAll {len(tests)} tests passed.")
