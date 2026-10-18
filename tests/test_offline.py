"""Offline tests: no API key or internet needed. Run:  python tests/test_offline.py"""
import json, os, sys, tempfile
from pathlib import Path
from types import SimpleNamespace

TMP = tempfile.mkdtemp()
os.environ["LOG_DIR"] = TMP          # keep test logs out of the real logs/ folder
os.environ["VERBOSE"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # noqa: E402


def lines(name):
    p = Path(TMP) / name
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []


class FakeClient:
    """Returns scripted responses in order, so the loop can be tested without a model."""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def resp(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


def call(cid, name, args):
    return SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def use(client):
    app._client = client
    app.get_client = lambda: client
    app.FALLBACKS = []          # keep the tests offline: never reach the real fallback providers


def test_sizing_matches_pdf_example():
    r = app.estimate_system_size([
        {"name": "refrigerator", "watts": 150, "hours_per_day": 8},
        {"name": "lighting", "watts": 100, "hours_per_day": 5},
        {"name": "television", "watts": 100, "hours_per_day": 4},
        {"name": "washing machine", "watts": 500, "hours_per_day": 1},
        {"name": "air conditioner", "watts": 1000, "hours_per_day": 5}], area="Beirut")
    assert r["daily_energy_kwh"] == 7.6, r
    assert r["required_array_kw"] == 1.9 and r["panels_450w"] == 5 and r["array_kwp"] == 2.25, r
    assert r["inverter_kw"] == 5 and r["battery_modules_5_12kwh"] == 1, r
    assert r["closest_illustrative_package"] == "Smart Home Hybrid", r


def test_sizing_uses_typical_values_and_area():
    r = app.estimate_system_size([{"name": "fridge"}, {"name": "AC"}, {"name": "LED TV"}], area="Zahle")
    assert r["peak_sun_hours"] == 5.5 and r["area_used"] == "Bekaa"
    assert any("assumed" in a for a in r["assumed_values"])
    r2 = app.estimate_system_size([{"name": "fridge"}], area="Narnia")
    assert r2["peak_sun_hours"] == 5.0 and "area_note" in r2 and r2["inverter_kw"] == 3


def test_sizing_edge_cases():
    assert app.estimate_system_size([{"name": "mystery gadget"}])["status"] == "needs_input"
    heater = app.estimate_system_size([{"name": "electric water heater"}, {"name": "oven", "watts": 2500, "hours_per_day": 1}])
    assert heater["needs_engineer_review"] is True
    big = app.estimate_system_size([{"name": "ac", "quantity": 12}], area="Beirut")
    assert big["needs_engineer_review"] and big["closest_illustrative_package"].startswith("Custom")


def test_lead_tool():
    bad = app.record_customer_interest("not-an-email", "Rami", "quote")
    assert bad["status"] == "error" and bad["error"] == "invalid_email"
    assert app.record_customer_interest("rami@example.com", "", "quote")["error"] == "missing_name"
    ok = app.record_customer_interest("rami@example.com", "Rami", "flat in Beirut, ~7.6 kWh/day")
    assert ok["status"] == "ok"
    assert lines("leads.jsonl")[-1]["email"] == "rami@example.com"


def test_feedback_tool_and_run_tool():
    assert app.record_feedback("Can I export to the grid?")["status"] == "ok"
    assert lines("feedback.jsonl")[-1]["question"] == "Can I export to the grid?"
    assert json.loads(app.run_tool("record_feedback", "{bad json"))["status"] == "error"
    assert json.loads(app.run_tool("nope", "{}"))["status"] == "error"


def test_router_rules():
    app.ROUTER_USE_LLM = False
    cases = {
        "I want a quote for my house": "lead",
        "my email is a@b.co": "lead",
        "Can you book a site survey for me?": "lead",
        "Can I sell power back to the grid?": "feedback",
        "How much does it cost?": "feedback",
        "My inverter stopped working": "feedback",
        "What warranty do the batteries have?": "general",
        "How many panels do I need?": "general",
    }
    for text, want in cases.items():
        assert app.route_message(text) == want, (text, app.route_message(text))


def test_router_llm_fallback():
    app.ROUTER_USE_LLM = True
    use(FakeClient([resp("lead")]))
    assert app.route_message("I'd love to go ahead with this") == "lead"
    use(FakeClient([resp("I think it is general.")]))
    assert app.route_message("what about winter?") == "general"
    app.ROUTER_USE_LLM = False


def test_history_normalisation():
    h = app.normalize_history([["hi", "hello"], {"role": "user", "content": [{"type": "text", "text": "x"}]}])
    assert h == [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
                 {"role": "user", "content": "x"}]
    assert len(app.normalize_history([{"role": "user", "content": "m"}] * 40)) == app.HISTORY_WINDOW


def test_agent_loop_tool_call_then_reflection_revises():
    app.ROUTER_USE_LLM = False
    app.REFLECTION_ENABLED = True
    fake = FakeClient([
        resp(None, [call("c1", "record_customer_interest",
                         {"email": "sara@example.com", "name": "Sara", "message": "villa quote"})]),
        resp("Done! A specialist will call you. A system like that usually costs around $8,000."),   # draft with a price
        resp(json.dumps({"verdict": "revise", "issues": ["mentions a price"],
                         "revised_reply": "Done! A specialist will call you within 1 business day. Anything else?"})),
    ])
    use(fake)
    out = app.run_agent("I want a quote for my villa. I'm Sara, sara@example.com", [])
    assert out["route"] == "lead" and out["tools_used"] == ["record_customer_interest"]
    assert "$" in out["draft"] and "$" not in out["reply"]
    # the lead tool received the right arguments and the tool result was fed back to the model
    assert lines("leads.jsonl")[-1]["name"] == "Sara"
    second_call_msgs = fake.calls[1]["messages"]
    assert second_call_msgs[-1]["role"] == "tool" and second_call_msgs[-2]["tool_calls"][0]["id"] == "c1"
    rec = lines("reflection.jsonl")[-1]
    assert rec["changed"] is True and any("price" in f for f in rec["code_flags"])
    # intent-specific tools were offered
    assert {t["function"]["name"] for t in fake.calls[0]["tools"]} == set(app.INTENT_TOOLS["lead"])


def test_feedback_safeguard_logs_when_model_forgets():
    app.REFLECTION_ENABLED = False
    use(FakeClient([resp("Thanks, I'm sorry about that. Could you share your email so the team can call you?")]))
    before = len(lines("feedback.jsonl"))
    out = app.run_agent("I'm unhappy, my battery stopped working", [])
    assert out["route"] == "feedback" and "record_feedback (safeguard)" in out["tools_used"]
    assert len(lines("feedback.jsonl")) == before + 1


def test_reflection_bad_json_keeps_draft_and_errors_are_friendly():
    app.REFLECTION_ENABLED = True
    use(FakeClient([resp("A 5 kWp system is typical."), resp("not json at all")]))
    out = app.run_agent("What is a kWp?", [])
    assert out["reply"] == out["draft"] == "A 5 kWp system is typical."
    def boom(**kw): raise ConnectionError("offline")
    use(SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=boom))))
    assert "technical problem" in app.agent_reply("hello", [])


def test_system_prompt_contents():
    p = app.SYSTEM_PROMPT
    assert "=== PERSONA ===" in p and "=== INSTRUCTIONS ===" in p and "WORKED EXAMPLE" in p
    assert "CedarSun" in p and "Smart Home Hybrid" in p
    assert "Never state, estimate or hint at prices" in p


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print("PASS", name)
    print(f"\nAll {len(tests)} tests passed.")
