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
from dataclasses import dataclass
from typing import Any

from . import config, clinical_rules

# İZİNLİ klinik alanlar (allowlist) — kimlik/geçmiş ASLA prompta girmez (F22).
ALLOWED_FIELDS = {
    "organ", "goruntuleme", "slice_thickness_mm", "pixel_spacing_mm",
    "lezyon_sayisi", "toplam_tumor_hacim_cm3",
    "en_buyuk_cap_lezyon", "en_buyuk_hacim_lezyon",
    "damar", "boyut_kategorisi", "vaskuler",
    "schema_version", "model", "model_agirlik", "damar_model", "provenance",
}

SYSTEM_PROMPT = (
    "Sen deneyimli bir abdominal radyologsun. Sana verilen BT segmentasyon "
    "bulgularını raporla.\n"
    "KURALLAR:\n"
    "- YALNIZ verilen metriklere dayan. Metrikte olmayan hiçbir sayı/bulgu UYDURMA.\n"
    "- Değeri 'null' veya 'değerlendirilemez' olan alanları ölçülmüş gibi sunma; "
    "eksikliği açıkça belirt.\n"
    "- Boyut kategorisi verildiyse bunu 'boyut kategorisi' olarak adlandır; "
    "kesin T-evresi olarak sunma.\n"
    "- Vasküler sarma araştırma ölçümüdür; tek başına kesin rezektabilite "
    "kararı verme, histopatolojik doğrulama öner.\n"
    "- Türkçe, kısa ve öz, düz metin yaz (Markdown kullanma).\n"
    "YAPI: 1.TEKNİK 2.BULGULAR 3.VASKÜLER (varsa) 4.BOYUT KATEGORİSİ (varsa) "
    "5.SONUÇ VE ÖNERİ"
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
    temiz = anonymize_findings(findings)
    return ("BT Segmentasyon Bulguları:\n"
            + json.dumps(temiz, ensure_ascii=False, indent=2)
            + "\n\nStandart radyoloji raporu oluştur.")


# ---------------------------------------------------------------------------
# Kural-tabanlı şablon (set 6) — %100 sadık baseline, API gerektirmez
# ---------------------------------------------------------------------------
def template_report(findings: dict[str, Any]) -> str:
    organ = findings.get("organ", "organ")
    n = findings.get("lezyon_sayisi", 0)
    enb = findings.get("en_buyuk_cap_lezyon") or {}
    satir = ["1. TEKNİK", f"{findings.get('goruntuleme', 'bilinmiyor')} inceleme.", ""]
    satir.append("2. BULGULAR")
    if n == 0:
        satir.append(f"{organ.capitalize()} parankiminde ölçülebilir fokal lezyon saptanmadı.")
    else:
        satir.append(f"{organ.capitalize()} parankiminde {n} adet lezyon izlendi.")
        if enb.get("cap_mm") is not None:
            satir.append(f"En büyük lezyonun aksiyel çapı {enb['cap_mm']} mm.")
        if enb.get("hacim_cm3") is not None:
            satir.append(f"Bu lezyonun hacmi {enb['hacim_cm3']} cm3.")
        tot = findings.get("toplam_tumor_hacim_cm3")
        if tot:
            satir.append(f"Toplam tümör hacmi {tot} cm3.")
    # Vasküler
    vask = findings.get("vaskuler", {}).get("damarlar", [])
    if vask:
        satir += ["", "3. VASKÜLER"]
        for v in vask:
            if v.get("durum") == "değerlendirilemez":
                satir.append(f"{v['damar']}: değerlendirilemedi.")
                continue
            m = v.get("mesafe_mm"); d = v.get("sarma_derece")
            m_s = "ölçülemedi" if m is None else f"{m} mm"
            d_s = "ölçülemedi" if d is None else f"{d} derece"
            satir.append(f"{v['damar']}: mesafe {m_s}, sarma {d_s} (araştırma ölçümü).")
    # Boyut kategorisi
    bk = findings.get("boyut_kategorisi", {})
    if bk.get("kategori"):
        satir += ["", "4. BOYUT KATEGORİSİ", f"Boyut kategorisi: {bk['kategori']}. {bk.get('uyari','')}"]
    satir += ["", "5. SONUÇ VE ÖNERİ",
              "Bulgular segmentasyon ölçümlerine dayanır; histopatolojik doğrulama önerilir."]
    return "\n".join(satir)


# ---------------------------------------------------------------------------
# Sağlayıcı adaptörleri (F19)
# ---------------------------------------------------------------------------
MODELS = {
    "Claude":   {"provider": "anthropic", "model": "claude-opus-4-8"},   # Opus/Sonnet kararı: config
    "Gemini":   {"provider": "google", "model": "gemini-2.5-flash"},
    "Qwen":     {"provider": "local_hf", "model": "Qwen2.5-14B-Instruct"},
    "Gemma":    {"provider": "local_hf", "model": "gemma-2-9b-it"},
    "Llama":    {"provider": "local_hf", "model": "Llama-3.1-8B-Instruct"},
    "Template": {"provider": "template", "model": "rule-based"},
}


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
             temperature: float = 0.3, max_tokens: int = 2500) -> LLMResponse:
    if provider == "template":
        return LLMResponse(template_report(findings or {}), provider, model,
                           finish_reason="stop")

    if provider == "anthropic":
        key = config.get_api_key("anthropic")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY yok (.env).")
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        # Opus/Sonnet reasoning: temperature API varsayılanıyla (gönderme).
        msg = client.messages.create(
            model=model, max_tokens=max_tokens, system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")
        return LLMResponse(text, provider, model, finish_reason=msg.stop_reason,
                           usage={"in": msg.usage.input_tokens, "out": msg.usage.output_tokens},
                           truncated=(msg.stop_reason == "max_tokens"))

    if provider == "google":
        key = config.get_api_key("google")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY yok (.env).")
        from google import genai
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model=model, contents=user_prompt,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt, temperature=temperature,
                max_output_tokens=max_tokens))          # F19: max_tokens aktarıldı
        fr = None
        try:
            fr = str(resp.candidates[0].finish_reason)
        except Exception:
            pass
        return LLMResponse(resp.text or "", provider, model, finish_reason=fr,
                           truncated=(fr is not None and "MAX_TOKENS" in fr.upper()))

    if provider == "local_hf":
        raise NotImplementedError(
            f"{model} açık model Colab (4-bit) ortamında çalıştırılır; "
            "ortak şema/prompt aynıdır.")

    raise ValueError(f"bilinmeyen sağlayıcı: {provider}")
