from google import genai
import os

os.environ['GOOGLE_API_KEY'] = 'AIzaSyCtorKFkpOfALcL6Jwz_PSQE1dR_6ZH-Xk'

client = genai.Client(api_key=os.environ['GOOGLE_API_KEY'])

rapor = """TEKNİK
Portal venöz faz BT incelemesi yapılmıştır. Kesit kalınlığı 2.5 mm, piksel aralığı 0.64 mm'dir.

BULGULAR
Pankreas hacmi 41.89 cm3. Pankreas başında 37.3 mm solid lezyon, hacmi 27.12 cm3.

VASKÜLER DEĞERLENDİRME
Portal ven ile temas mevcut, 293° sarma. Aorta mesafesi 2.32 mm, sarma yok.

T EVRELEMESİ VE REZEKTABİLİTE
T2 evre. Portal ven 293° sarma nedeniyle lokal ileri/irresektabl.

SONUÇ VE ÖNERİ
Pankreas başında T2 evre lokal ileri lezyon. Histopatolojik doğrulama önerilir."""

system = f"""Sen 15 yıllık deneyimli bir abdominal radyologsun.
Aşağıdaki raporu sen yazdın. Sorulara kısa, net ve klinik dilde cevap ver.
Bilmediğin şeyleri uydurmadan 'bu veri mevcut değil' de. Türkçe cevap ver.
RAPOR:
{rapor}"""

print("=" * 50)
print("RADYOLOJİ SORU-CEVAP SİSTEMİ")
print("Çıkmak için 'q' yazın")
print("=" * 50)

while True:
    soru = input("\nSorunuz: ")
    if soru.lower() == 'q':
        break
    cevap = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=soru,
        config=genai.types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.3
        )
    )
    print(f"\nRadyolog: {cevap.text}")