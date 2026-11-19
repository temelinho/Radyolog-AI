"""
reporting.py — ortak şema, prompt, sağlayıcı adaptörleri ve şablon (F07, F19, F22).

Düzeltmeler:
  * Tek rapor şeması (schema_version); LLM girdisi ile değerlendirme girdisi
    AYNI JSON.
  * Kimlik filtresi KARA LİSTE değil, İZİNLİ ALAN şeması (F22). Boş bulguda
    tüm kayda geri düşme YOK.
  * Sağlayıcı yanıtı finish_reason/token ile kaydedilir; boş/kesik yanıt
    BAŞARI sayılmaz (F19). Google çağrısına max_tokens aktarılır (F19 bug'ı).
  * Kural-tabanlı şablon: %100 sadık kontrol (baseline); API gerektirmez.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from . import config, clinical_rules

# İZİNLİ klinik alanlar (allowlist) — kimlik/geçmiş ASLA prompta girmez (F22).
ALLOWED_FIELDS = {
    "organ", "goruntuleme", "slice_thickness_mm", "pixel_spacing_mm",
    "lezyon_sayisi", "toplam_tumor_hacim_cm3",
    "en_buyuk_cap_lezyon", "en_buyuk_hacim_lezyon",
    "damar", "boyut_kategorisi", "vaskuler",
    # NOT: Modele YALNIZ organ + klinik metrikler gider. Veri seti / segmentasyon
    # modeli kimliği (model, model_agirlik, damar_model) ve teknik meta
    # (schema_version, provenance) BİLEREK yok — kaynak sızmasın, model yanlı
    # olmasın; provenance tam bulgular.json'da yerelde kalır (F22, RUA).
}

SYSTEM_PROMPT = (
    "You are an experienced abdominal radiologist. Report the CT segmentation "
    "findings you are given.\n"
    "RULES:\n"
    "- Rely ONLY on the provided metrics. Do NOT invent any number or finding "
    "that is not in the metrics.\n"
    "- Do not present a field whose value is 'null' or 'not assessed' as if it "
    "were measured; state the absence explicitly.\n"
    "- If a size category is given, call it a 'size category'; do not present it "
    "as a definitive T-stage.\n"
    "- Vascular encasement is a research-grade measurement; do not decide "
    "resectability from it alone, and recommend histopathological confirmation.\n"
    "- Write in English as a complete, standard radiology report in plain text "
    "(no Markdown). Be as thorough as the metrics support; do not pad with "
    "content that is not grounded in the given metrics.\n"
    "STRUCTURE: 1. TECHNIQUE 2. FINDINGS 3. VASCULAR (if any) 4. SIZE CATEGORY "
    "(if any) 5. IMPRESSION AND RECOMMENDATION"
)


# ---------------------------------------------------------------------------
# Şema oluşturma
# ---------------------------------------------------------------------------
def build_findings(*, organ: str, summary: dict[str, Any],
                   vessels: dict[str, Any] | None = None,
                   acquisition: dict[str, Any] | None = None,
                   provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    """Ölçüm + kuraldan schema v2 bulgular.json üretir."""
    enb = summary.get("en_buyuk_cap_lezyon")
    dia = enb["cap_mm"] if isinstance(enb, dict) else None
    findings: dict[str, Any] = {
        "schema_version": config.SCHEMA_VERSION,
        "organ": organ,
        "lezyon_sayisi": summary.get("lezyon_sayisi", 0),
        "toplam_tumor_hacim_cm3": summary.get("toplam_tumor_hacim_cm3", 0.0),
        "en_buyuk_cap_lezyon": enb,
        "en_buyuk_hacim_lezyon": summary.get("en_buyuk_hacim_lezyon"),
        "damar": vessels or {},
        "boyut_kategorisi": clinical_rules.size_category(dia, organ),
        "vaskuler": clinical_rules.vascular_assessment(vessels or {}, organ),
        "goruntuleme": (acquisition or {}).get("faz", "bilinmiyor"),  # F06: metadata yoksa bilinmiyor
        "provenance": provenance or {},
    }
    return findings


def anonymize_findings(findings: dict[str, Any]) -> dict[str, Any]:
    """İzinli alan şemasıyla temizle (F22). Bilinmeyen/kimlik alanları DÜŞER."""
    return {k: v for k, v in findings.items() if k in ALLOWED_FIELDS}


def model_findings(findings: dict[str, Any]) -> dict[str, Any]:
    """Canonical English clinical JSON shared by all report generators."""
    f = anonymize_findings(findings)
    organ = config.organ_en(f.get("organ"))
    if organ not in ("liver", "pancreas"):
        raise ValueError(f"Unsupported reporting organ: {organ!r}")
    phase = (f.get("goruntuleme") or "").strip().lower()
    phases = {
        "": None, "bilinmiyor": None, "unknown": None, "not specified": None,
        "portal venöz faz bt": "portal venous",
        "portal venöz": "portal venous", "portal venous": "portal venous",
        "arteriyel": "arterial", "arterial": "arterial",
        "kontrastsız": "noncontrast", "noncontrast": "noncontrast",
        "geç faz": "delayed", "delayed": "delayed",
    }
    if phase not in phases:
        raise ValueError(f"Unrecognized imaging phase: {phase!r}")

    def lesion(value):
        if not isinstance(value, dict):
            return None
        return {"lesion_id": value.get("id"),
                "axial_diameter_mm": value.get("cap_mm"),
                "volume_cm3": value.get("hacim_cm3")}

    vessels = {}
    for name, value in sorted((f.get("damar") or {}).items()):
        if name not in config.VESSEL_EN:
            raise ValueError(f"Unrecognized vessel: {name!r}")
        vessels[config.vessel_en(name)] = {
            "distance_mm": value.get("mesafe_mm"),
            "encasement_angle_deg": value.get("sarma_derece"),
            "status": value.get("status", "not_assessed"
                                if value.get("mesafe_mm") is None
                                and value.get("sarma_derece") is None else "success"),
        }
    return {
        "organ": organ,
        "imaging_phase": phases[phase],
        "lesion_count": f.get("lezyon_sayisi"),
        "total_tumor_volume_cm3": f.get("toplam_tumor_hacim_cm3"),
        "largest_lesion_by_diameter": lesion(f.get("en_buyuk_cap_lezyon")),
        "largest_lesion_by_volume": lesion(f.get("en_buyuk_hacim_lezyon")),
        "vessels": vessels,
        "size_category": (f.get("boyut_kategorisi") or {}).get("kategori"),
    }


def build_prompt(findings: dict[str, Any]) -> str:
    """Serialize identical English clinical JSON for every model."""
    payload = json.dumps(model_findings(findings), ensure_ascii=False,
                         sort_keys=True, indent=2, allow_nan=False)
    return ("CT segmentation metrics (JSON):\n" + payload
            + "\n\nWrite a standard radiology report in English.")


def request_identity(provider: str, model: str, system_prompt: str,
                     user_prompt: str) -> dict[str, Any]:
    """Bind cached reports to the exact input and recorded settings."""
    import hashlib
    conditions = generation_conditions(provider, model)
    text = build_input(system_prompt, user_prompt)
    encoded = json.dumps({"input": text, "conditions": conditions},
                         sort_keys=True, ensure_ascii=False).encode("utf-8")
    return {"request_sha256": hashlib.sha256(encoded).hexdigest(),
            "input_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "conditions": conditions}


# ---------------------------------------------------------------------------
# Kural-tabanlı şablon (set 6) — %100 sadık baseline, API gerektirmez
# ---------------------------------------------------------------------------
def template_report(findings: dict[str, Any]) -> str:
    """Kural tabanlı KONTROL raporu. BDM değildir; üretim mekanizması farklıdır.
    Yalnız modellere giden İngilizce klinik JSON'u (model_findings) kullanır,
    böylece LLM'lerle AYNI klinik değerleri görür. Eksik değer (null) asla
    "yok/0" olarak yazılmaz; "not assessed" yazılır."""
    c = model_findings(findings)
    organ = c["organ"]
    n = c["lesion_count"]
    big = c["largest_lesion_by_diameter"] or {}
    phase = c["imaging_phase"]
    teknik = ("CT study, phase not specified." if phase is None
              else f"{phase.capitalize()} phase CT study.")
    L = ["1. TECHNIQUE", teknik, ""]
    L.append("2. FINDINGS")
    if n is None:
        L.append(f"Lesion count in the {organ} was not assessed.")
    elif n == 0:
        L.append(f"No measurable focal lesion detected in the {organ} parenchyma.")
    else:
        L.append(f"{n} lesion(s) observed in the {organ} parenchyma.")
        if big.get("axial_diameter_mm") is not None:
            L.append(f"The largest lesion has an axial diameter of {big['axial_diameter_mm']} mm.")
        if big.get("volume_cm3") is not None:
            L.append(f"The volume of this lesion is {big['volume_cm3']} cm3.")
        tot = c["total_tumor_volume_cm3"]
        if tot is not None:
            L.append(f"Total tumor volume is {tot} cm3.")
    if c["vessels"]:
        L += ["", "3. VASCULAR"]
        for name, v in c["vessels"].items():
            m, d = v.get("distance_mm"), v.get("encasement_angle_deg")
            if m is None and d is None:
                L.append(f"{name}: not assessed.")
                continue
            m_s = "not measured" if m is None else f"{m} mm"
            d_s = "not measured" if d is None else f"{d} degrees"
            L.append(f"{name}: distance {m_s}, encasement {d_s} (research-grade).")
    if c["size_category"]:
        L += ["", "4. SIZE CATEGORY",
              f"Size category: {c['size_category']} (size only, not a T-stage)."]
    L += ["", "5. IMPRESSION AND RECOMMENDATION",
          "Findings are based on segmentation measurements; histopathological "
          "confirmation is recommended."]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Sağlayıcı adaptörleri (F19)
# ---------------------------------------------------------------------------
MODELS = {
    "Claude":   {"provider": "anthropic", "model": "claude-sonnet-5-5"},  # Sonnet (bütçe); diğerleriyle aynı koşul
    "Gemini":   {"provider": "google", "model": "gemini-3.8-flash"},  # 2.5-flash yeni kullanicilara kapandi
    "Qwen":     {"provider": "local_hf", "model": "Qwen2.5-14B-Instruct"},
    "Gemma":    {"provider": "local_hf", "model": "gemma-2-9b-it"},
    "Llama":    {"provider": "local_hf", "model": "Llama-3.1-8B-Instruct"},
    "Template": {"provider": "template", "model": "rule-based"},
}

# Açık model kısa adı -> HuggingFace repo kimliği (Colab 4-bit çıkarımı).
# Llama-3.1 ve gemma-2 HF'de KAPILIDIR: lisans kabulü + `huggingface_hub.login`
# gerekir. Qwen2.5 açıktır.
HF_REPOS = {
    "Qwen2.5-14B-Instruct": "Qwen/Qwen2.5-14B-Instruct",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
}

# Yüklenen model+tokenizer burada tutulur: 80 vakada TEKRAR TEKRAR yüklenmez.
# Dikkat: her seferinde TEK açık model önbellekte olmalı (T4 16 GB VRAM); farklı
# bir açık modele geçmeden önce `_HF_CACHE.clear()` + `torch.cuda.empty_cache()`.
_HF_CACHE: dict[str, Any] = {}


def _load_hf(model: str):
    """HF modelini 4-bit yükle (bir kez), (tokenizer, model) döndür."""
    if model in _HF_CACHE:
        return _HF_CACHE[model]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    repo = HF_REPOS.get(model, model)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(repo)
    mdl = AutoModelForCausalLM.from_pretrained(
        repo, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16)
    _HF_CACHE[model] = (tok, mdl)
    return tok, mdl


def free_hf_cache() -> None:
    """Açık model VRAM'ini boşalt (modeller arası geçişte çağrılır)."""
    _HF_CACHE.clear()
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    truncated: bool = False

    def is_valid(self) -> bool:
        """Boş veya kesilmiş yanıt BAŞARI değildir (F19)."""
        return bool(self.text and self.text.strip()) and not self.truncated


# ---------------------------------------------------------------------------
# ÜRETİM KOŞULLARI — TÜM MODELLER İÇİN TEK KAYNAK.
# Claude Sonnet 5.5 örnekleme parametrelerini (temperature/top_p/top_k) kabul
# etmez ve düşünmesi yalnız 'between_tools' ile kapatılabilir. Eşitlik bu yüzden
# Ortak metin ve çıktı bütçesi aşağıda tanımlıdır:
#   * Girdi: herkese BİREBİR AYNI tek kullanıcı mesajı (sistem istemi + vaka).
#   * Düşünme: hepsinde KAPALI (Claude: between_tools; açık modellerde yok).
#   * Örnekleme eşdeğerliği bilinmiyor. Claude: API varsayılanı (parametre
#     gönderilmez). Açık modeller: temperature 1.0, top_p 1.0, top_k kapalı,
#     repetition_penalty 1.0 — modelin kendi generation_config varsayılanları EZİLİR.
#   * Uzunluk sınırı: hepsinde aynı.
# ---------------------------------------------------------------------------
GEN = {
    "max_tokens": 3500,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,                 # 0 = top-k filtresi yok
    "repetition_penalty": 1.0,
    "seed": int(os.environ.get("RADYOLOG_SEED", "42")),
}


def build_input(system_prompt: str, user_prompt: str) -> str:
    """Her modele giden TEK metin. Tüm sağlayıcılar bunu kullanır."""
    return system_prompt.strip() + "\n\n" + user_prompt.strip()


def generation_conditions(provider: str, model: str) -> dict[str, Any]:
    """Bu sağlayıcıya GERÇEKTE uygulanan koşullar (kayıt/denetim için)."""
    base = {"provider": provider, "model": model,
            "input": "single user message = system prompt + case prompt",
            "max_tokens": GEN["max_tokens"], "thinking": "off",
            "input_format": "English clinical JSON",
            "prompt_version": config.REPORT_PROMPT_VERSION,
            "sampling_equivalence": "not established across providers"}
    if provider == "anthropic":
        base.update({"sampling": "API default (temperature/top_p/top_k not settable)",
                     "thinking_param": {"type": "between_tools"}, "seed": None,
                     "precision": "not disclosed by provider"})
    elif provider == "local_hf":
        base.update({"sampling": "unmodified distribution",
                     "temperature": GEN["temperature"], "top_p": GEN["top_p"],
                     "top_k": GEN["top_k"], "repetition_penalty": GEN["repetition_penalty"],
                     "seed": GEN["seed"], "quantization": "4-bit NF4",
                     "precision": "4-bit NF4 weights (bitsandbytes), fp16 compute"})
    elif provider == "google":
        base.update({"sampling": "unmodified distribution",
                     "temperature": GEN["temperature"], "top_p": GEN["top_p"]})
    elif provider == "template":
        base.update({"sampling": "deterministic (rule-based)", "thinking": "n/a"})
    return base


def call_llm(provider: str, model: str, system_prompt: str, user_prompt: str,
             findings: dict[str, Any] | None = None) -> LLMResponse:
    """Use shared input and record provider-specific generation settings."""
    if provider == "template":
        return LLMResponse(template_report(findings or {}), provider, model,
                           finish_reason="stop")

    text_in = build_input(system_prompt, user_prompt)
    max_tokens = GEN["max_tokens"]

    if provider == "anthropic":
        key = config.get_api_key("anthropic")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY yok (.env).")
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        msg = client.messages.create(
            model=model, max_tokens=max_tokens,
            thinking={"type": "between_tools"},          # düşünme KAPALI
            messages=[{"role": "user", "content": text_in}])
        if msg.stop_reason == "refusal":
            return LLMResponse("", provider, model, finish_reason="refusal")
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        thought = any(getattr(b, "type", None) in ("thinking", "redacted_thinking")
                      for b in msg.content)
        return LLMResponse(text, provider, model, finish_reason=msg.stop_reason,
                           usage={"in": msg.usage.input_tokens, "out": msg.usage.output_tokens,
                                  "thinking_blocks": thought},
                           truncated=(msg.stop_reason == "max_tokens"))

    if provider == "google":
        key = config.get_api_key("google")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY yok (.env).")
        from google import genai
        import re as _re, time as _time
        client = genai.Client(api_key=key)
        gen_cfg = genai.types.GenerateContentConfig(
            temperature=GEN["temperature"], top_p=GEN["top_p"],
            max_output_tokens=max_tokens)
        # Ücretsiz kota (5 istek/dk) ve geçici 503 için retry+backoff.
        tries = int(os.environ.get("RADYOLOG_GEMINI_RETRIES", "6") or 6)
        resp = None
        for _att in range(tries):
            try:
                resp = client.models.generate_content(
                    model=model, contents=text_in, config=gen_cfg)
                break
            except Exception as _e:
                msg = str(_e)
                transient = ("429" in msg or "RESOURCE_EXHAUSTED" in msg
                             or "503" in msg or "UNAVAILABLE" in msg)
                daily = ("PerDay" in msg or "per day" in msg.lower())
                m = _re.search(r"retry(?:Delay)?['\"]?\s*[:=]?\s*['\"]?\s*"
                               r"(?:in\s*)?([0-9]+(?:\.[0-9]+)?)\s*s", msg, _re.I)
                wait = float(m.group(1)) if m else 0.0
                if not transient or daily or wait > 120 or _att == tries - 1:
                    raise
                wait = max(wait, 13.0 * (_att + 1))
                _time.sleep(min(wait, 65.0))
        fr = None
        try:
            fr = str(resp.candidates[0].finish_reason)
        except Exception:
            pass
        return LLMResponse(resp.text or "", provider, model, finish_reason=fr,
                           truncated=(fr is not None and "MAX_TOKENS" in fr.upper()))

    if provider == "local_hf":
        import torch
        tok, mdl = _load_hf(model)
        user_only = [{"role": "user", "content": text_in}]
        try:
            # Boş sistem mesajı: bazı şablonların (ör. Qwen) kendi varsayılan
            # sistem metnini ("You are Qwen ...") EKLEMESİNİ engeller.
            enc = tok.apply_chat_template([{"role": "system", "content": ""}] + user_only,
                                          add_generation_prompt=True, return_tensors="pt")
        except Exception:
            # Sistem rolünü desteklemeyen şablonlar (ör. gemma-2): yalnız kullanıcı.
            enc = tok.apply_chat_template(user_only, add_generation_prompt=True,
                                          return_tensors="pt")
        # Yeni transformers 'BatchEncoding' döndürür (tensor DEĞİL): input_ids'i ayıkla.
        if isinstance(enc, torch.Tensor):
            input_ids = enc.to(mdl.device)
            gen_kwargs = {"input_ids": input_ids}
        else:
            input_ids = enc["input_ids"].to(mdl.device)
            gen_kwargs = {"input_ids": input_ids}
            if "attention_mask" in enc:
                gen_kwargs["attention_mask"] = enc["attention_mask"].to(mdl.device)
        in_len = input_ids.shape[-1]
        torch.manual_seed(GEN["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(GEN["seed"])
        with torch.no_grad():
            out = mdl.generate(**gen_kwargs, max_new_tokens=max_tokens,
                               do_sample=True,
                               temperature=GEN["temperature"], top_p=GEN["top_p"],
                               top_k=GEN["top_k"],
                               repetition_penalty=GEN["repetition_penalty"],
                               pad_token_id=tok.eos_token_id)
        gen = out[0][in_len:]
        text = tok.decode(gen, skip_special_tokens=True)
        truncated = int(gen.shape[-1]) >= max_tokens  # yeni token sınıra dayandıysa kesik
        return LLMResponse(text, provider, model,
                           finish_reason="length" if truncated else "stop",
                           usage={"in": int(in_len), "out": int(gen.shape[-1])},
                           truncated=truncated)

    raise ValueError(f"bilinmeyen sağlayıcı: {provider}")
