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


def build_prompt(findings: dict[str, Any]) -> str:
    """Modele giden istem: YALNIZ organ + klinik sayısal metrikler, İngilizce.

    Veri seti / segmentasyon modeli kimliği ve teknik meta VERİLMEZ (modele kaynak
    sızdırmamak, yanlılık yaratmamak). Türkçe kategorik uyarı metinleri yerine
    yalnız sayısal değerler sunulur; klinik çerçeveyi sistem promptu verir.
    """
    f = anonymize_findings(findings)   # güvenlik: yalnız izinli (klinik) alanlar
    L = ["CT segmentation findings — use ONLY these values, invent nothing:"]
    L.append(f"- Organ: {f.get('organ', 'unknown')}")
    n = f.get("lezyon_sayisi")
    L.append(f"- Number of lesions: {'not assessed' if n is None else n}")
    tv = f.get("toplam_tumor_hacim_cm3")
    if tv is not None:
        L.append(f"- Total tumor volume: {tv} cm3")
    enb = f.get("en_buyuk_cap_lezyon")
    if isinstance(enb, dict):
        L.append(f"- Largest lesion by diameter: {enb.get('cap_mm')} mm diameter, "
                 f"{enb.get('hacim_cm3')} cm3 volume")
    env = f.get("en_buyuk_hacim_lezyon")
    if isinstance(env, dict):
        L.append(f"- Largest lesion by volume: {env.get('cap_mm')} mm diameter, "
                 f"{env.get('hacim_cm3')} cm3 volume")
    damar = f.get("damar") or {}
    if isinstance(damar, dict) and damar:
        L.append("- Vessels (research-grade):")
        for vn, vi in damar.items():
            if not isinstance(vi, dict):
                continue
            d = vi.get("mesafe_mm")
            a = vi.get("sarma_derece")
            ds = f"distance {d} mm" if d is not None else "distance not assessed"
            as_ = f"encasement angle {a} degrees" if a is not None else "encasement not assessed"
            L.append(f"    - {vn.replace('_', ' ')}: {ds}, {as_}")
    bk = f.get("boyut_kategorisi")
    if isinstance(bk, dict) and bk.get("kategori"):
        L.append(f"- Size category (size only, NOT a T-stage): {bk['kategori']}")
    return "\n".join(L) + "\n\nWrite a standard radiology report in English."


# ---------------------------------------------------------------------------
# Kural-tabanlı şablon (set 6) — %100 sadık baseline, API gerektirmez
# ---------------------------------------------------------------------------
def template_report(findings: dict[str, Any]) -> str:
    organ = findings.get("organ", "organ")
    n = findings.get("lezyon_sayisi", 0)
    enb = findings.get("en_buyuk_cap_lezyon") or {}
    faz = findings.get("goruntuleme") or "unknown"
    if faz in ("bilinmiyor", "unknown", ""):
        teknik = "CT study, phase not specified."
    else:
        teknik = f"{faz} CT study."
    L = ["1. TECHNIQUE", teknik, ""]
    L.append("2. FINDINGS")
    if not n:
        L.append(f"No measurable focal lesion detected in the {organ} parenchyma.")
    else:
        L.append(f"{n} lesion(s) observed in the {organ} parenchyma.")
        if enb.get("cap_mm") is not None:
            L.append(f"The largest lesion has an axial diameter of {enb['cap_mm']} mm.")
        if enb.get("hacim_cm3") is not None:
            L.append(f"The volume of this lesion is {enb['hacim_cm3']} cm3.")
        tot = findings.get("toplam_tumor_hacim_cm3")
        if tot:
            L.append(f"Total tumor volume is {tot} cm3.")
    # Vascular
    vask = (findings.get("vaskuler") or {}).get("damarlar", [])
    if vask:
        L += ["", "3. VASCULAR"]
        for v in vask:
            if v.get("durum") == "değerlendirilemez":
                L.append(f"{v['damar'].replace('_', ' ')}: not assessed.")
                continue
            m = v.get("mesafe_mm")
            d = v.get("sarma_derece")
            m_s = "not measured" if m is None else f"{m} mm"
            d_s = "not measured" if d is None else f"{d} degrees"
            L.append(f"{v['damar'].replace('_', ' ')}: distance {m_s}, "
                     f"encasement {d_s} (research-grade).")
    # Size category
    bk = findings.get("boyut_kategorisi", {})
    if bk.get("kategori"):
        L += ["", "4. SIZE CATEGORY",
              f"Size category: {bk['kategori']} (size only, not a T-stage)."]
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


def call_llm(provider: str, model: str, system_prompt: str, user_prompt: str,
             findings: dict[str, Any] | None = None,
             temperature: float = 0.3, max_tokens: int = 3500) -> LLMResponse:
    if provider == "template":
        return LLMResponse(template_report(findings or {}), provider, model,
                           finish_reason="stop")

    if provider == "anthropic":
        key = config.get_api_key("anthropic")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY yok (.env).")
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        # TÜM modeller AYNI koşul: temperature gönderilir. Bazı SDK/model sürümleri
        # temperature kabul etmezse (TypeError) onsuz tekrar denenir ki Claude çökmesin.
        base = dict(model=model, max_tokens=max_tokens, system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}])
        try:
            msg = client.messages.create(temperature=temperature, **base)
        except TypeError:
            msg = client.messages.create(**base)
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return LLMResponse(text, provider, model, finish_reason=msg.stop_reason,
                           usage={"in": msg.usage.input_tokens, "out": msg.usage.output_tokens},
                           truncated=(msg.stop_reason == "max_tokens"))

    if provider == "google":
        key = config.get_api_key("google")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY yok (.env).")
        from google import genai
        import re as _re, time as _time
        client = genai.Client(api_key=key)
        gen_cfg = genai.types.GenerateContentConfig(
            system_instruction=system_prompt, temperature=temperature,
            max_output_tokens=max_tokens)               # F19: max_tokens aktarıldı
        # Ücretsiz kota (5 istek/dk) ve geçici 503 için retry+backoff. Hata
        # 'retryDelay'/'retry in Ns' taşır; onu okuyup bekleriz, yoksa üstel backoff.
        tries = int(os.environ.get("RADYOLOG_GEMINI_RETRIES", "6") or 6)
        resp = None
        for _att in range(tries):
            try:
                resp = client.models.generate_content(
                    model=model, contents=user_prompt, config=gen_cfg)
                break
            except Exception as _e:
                msg = str(_e)
                transient = ("429" in msg or "RESOURCE_EXHAUSTED" in msg
                             or "503" in msg or "UNAVAILABLE" in msg)
                if not transient or _att == tries - 1:
                    raise
                m = _re.search(r"retry(?:Delay)?['\"]?\s*[:=]?\s*['\"]?\s*"
                               r"(?:in\s*)?([0-9]+(?:\.[0-9]+)?)\s*s", msg, _re.I)
                wait = float(m.group(1)) if m else 0.0
                wait = max(wait, 13.0 * (_att + 1))      # 5/dk için en az ~13s
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
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        try:
            enc = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt")
        except Exception:
            # Bazı şablonlar (ör. gemma-2) 'system' rolünü desteklemez: birleştir.
            merged = system_prompt + "\n\n" + user_prompt
            enc = tok.apply_chat_template(
                [{"role": "user", "content": merged}],
                add_generation_prompt=True, return_tensors="pt")
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
        with torch.no_grad():
            out = mdl.generate(**gen_kwargs, max_new_tokens=max_tokens,
                               do_sample=temperature > 0,
                               temperature=max(temperature, 1e-5),
                               pad_token_id=tok.eos_token_id)
        gen = out[0][in_len:]
        text = tok.decode(gen, skip_special_tokens=True)
        truncated = int(gen.shape[-1]) >= max_tokens  # yeni token sınıra dayandıysa kesik
        return LLMResponse(text, provider, model,
                           finish_reason="length" if truncated else "stop",
                           usage={"in": int(in_len), "out": int(gen.shape[-1])},
                           truncated=truncated)

    raise ValueError(f"bilinmeyen sağlayıcı: {provider}")
