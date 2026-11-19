"""TÜM modeller AYNI koşulda mı? Sağlayıcılara giden istekleri yakalayıp karşılaştırır.

Ağ/GPU kullanmaz: anthropic modülü ve HF model/tokenizer sahte nesnelerle değiştirilir.
Kontrol edilenler:
  1. Her sağlayıcının modele verdiği METİN birebir aynı.
  2. Hiçbir sağlayıcıya ayrı 'system' içeriği gitmiyor (yalnız tek kullanıcı mesajı).
  3. Uzunluk sınırı hepsinde aynı.
  4. Claude'da düşünme kapalı (between_tools) ve örnekleme parametresi gönderilmiyor.
  5. Açık modellerde örnekleme kesilmemiş dağılım (T=1, top_p=1, top_k=0, rp=1) ve seed sabit;
     modelin kendi generation_config varsayılanları açıkça EZİLİYOR.
"""

import sys
import types
import unittest

import torch

from pipeline import reporting as REP

FINDINGS = REP.build_findings(
    organ="karaciger",
    summary={"lezyon_sayisi": 1, "toplam_tumor_hacim_cm3": 8.4,
             "en_buyuk_cap_lezyon": {"id": 1, "cap_mm": 20.0, "hacim_cm3": 8.4},
             "en_buyuk_hacim_lezyon": {"id": 1, "cap_mm": 20.0, "hacim_cm3": 8.4},
             "status": "success"},
    vessels={"aorta": {"mesafe_mm": 28.4, "sarma_derece": 0}})
PROMPT = REP.build_prompt(FINDINGS)
EXPECTED_INPUT = REP.build_input(REP.SYSTEM_PROMPT, PROMPT)


# --- Sahte Anthropic -------------------------------------------------------
class _Block:
    type = "text"
    text = "report"


class _Usage:
    input_tokens = 10
    output_tokens = 5


class _Msg:
    content = [_Block()]
    stop_reason = "end_turn"
    usage = _Usage()


class _FakeMessages:
    def __init__(self, sink):
        self.sink = sink

    def create(self, **kwargs):
        self.sink.append(kwargs)
        return _Msg()


class _FakeAnthropic:
    sink: list = []

    def __init__(self, api_key=None):
        self.messages = _FakeMessages(_FakeAnthropic.sink)


# --- Sahte HF tokenizer + model ---------------------------------------------
class _FakeTok:
    eos_token_id = 0

    def __init__(self, accept_system=True):
        self.accept_system = accept_system
        self.seen = []

    def apply_chat_template(self, messages, add_generation_prompt, return_tensors):
        if not self.accept_system and any(m["role"] == "system" for m in messages):
            raise ValueError("System role not supported")
        self.seen.append(messages)
        return {"input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]])}

    def decode(self, ids, skip_special_tokens=True):
        return "report"


class _FakeModel:
    device = "cpu"

    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return torch.tensor([[1, 2, 3, 7, 8]])


class TestSameConditions(unittest.TestCase):
    def setUp(self):
        _FakeAnthropic.sink.clear()
        mod = types.ModuleType("anthropic")
        mod.Anthropic = _FakeAnthropic
        self._old_anthropic = sys.modules.get("anthropic")
        sys.modules["anthropic"] = mod
        self._old_key = REP.config.get_api_key
        REP.config.get_api_key = lambda name: "test-key"

    def tearDown(self):
        if self._old_anthropic is None:
            sys.modules.pop("anthropic", None)
        else:
            sys.modules["anthropic"] = self._old_anthropic
        REP.config.get_api_key = self._old_key
        REP._HF_CACHE.clear()

    def _run_local(self, accept_system):
        tok, mdl = _FakeTok(accept_system), _FakeModel()
        REP._HF_CACHE["fake-local"] = (tok, mdl)
        REP.call_llm("local_hf", "fake-local", REP.SYSTEM_PROMPT, PROMPT, findings=FINDINGS)
        return tok, mdl

    def test_claude_request(self):
        REP.call_llm("anthropic", "claude-sonnet-5-5", REP.SYSTEM_PROMPT, PROMPT,
                     findings=FINDINGS)
        req = _FakeAnthropic.sink[-1]
        self.assertEqual(req["messages"], [{"role": "user", "content": EXPECTED_INPUT}])
        self.assertNotIn("system", req)
        self.assertEqual(req["thinking"], {"type": "between_tools"})
        self.assertEqual(req["max_tokens"], REP.GEN["max_tokens"])
        for p in ("temperature", "top_p", "top_k"):
            self.assertNotIn(p, req)

    def test_local_models_same_text_and_sampling(self):
        for accept_system in (True, False):          # Qwen/Llama tipi ve Gemma tipi şablon
            tok, mdl = self._run_local(accept_system)
            msgs = tok.seen[-1]
            user = [m for m in msgs if m["role"] == "user"]
            system = [m for m in msgs if m["role"] == "system"]
            self.assertEqual(user, [{"role": "user", "content": EXPECTED_INPUT}])
            self.assertTrue(all(m["content"] == "" for m in system))   # boş sistem
            k = mdl.kwargs
            self.assertTrue(k["do_sample"])
            self.assertEqual(k["max_new_tokens"], REP.GEN["max_tokens"])
            self.assertEqual(k["temperature"], 1.0)
            self.assertEqual(k["top_p"], 1.0)
            self.assertEqual(k["top_k"], 0)
            self.assertEqual(k["repetition_penalty"], 1.0)

    def test_local_seed_reproducible(self):
        _, mdl = self._run_local(True)
        a = torch.rand(1).item()
        self._run_local(True)
        b = torch.rand(1).item()
        self.assertEqual(a, b)   # her çağrıda aynı tohumla başlıyor

    def test_conditions_record(self):
        c = {p: REP.generation_conditions(p, "m") for p in ("anthropic", "local_hf")}
        for v in c.values():
            self.assertEqual(v["thinking"], "off")
            self.assertEqual(v["max_tokens"], REP.GEN["max_tokens"])
            self.assertEqual(v["input"], "single user message = system prompt + case prompt")

    def test_claude_thinking_arm_is_separate(self):
        REP.call_llm("anthropic_thinking", "claude-sonnet-5-5", REP.SYSTEM_PROMPT, PROMPT,
                     findings=FINDINGS)
        req = _FakeAnthropic.sink[-1]
        self.assertEqual(req["messages"], [{"role": "user", "content": EXPECTED_INPUT}])
        self.assertEqual(req["thinking"], {"type": "adaptive"})
        self.assertEqual(req["max_tokens"], REP.THINKING_MAX_TOKENS)
        for p in ("temperature", "top_p", "top_k", "system"):
            self.assertNotIn(p, req)
        a = REP.request_identity("anthropic", "claude-sonnet-5-5", REP.SYSTEM_PROMPT, PROMPT)
        b = REP.request_identity("anthropic_thinking", "claude-sonnet-5-5", REP.SYSTEM_PROMPT, PROMPT)
        self.assertNotEqual(a["request_sha256"], b["request_sha256"])   # önbellekler karışmaz
        self.assertEqual(a["input_sha256"], b["input_sha256"])          # girdi aynı
        self.assertIn("Claude-thinking", REP.MODELS)


if __name__ == "__main__":
    unittest.main()
