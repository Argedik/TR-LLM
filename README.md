# Türkçe LLM Geliştirme Altyapısı

Bu projede Türkçe dil modeli (LLM) eğitimi için temel altyapı kurulmuştur.

## Gereksinimler
- Python 3.8+
- transformers
- datasets
- torch
- accelerate
- sentencepiece
- tqdm

## Kurulum
Aşağıdaki komut ile gerekli kütüphaneleri kurabilirsiniz:

```bash
pip install -r requirements.txt
```

## Klasör Yapısı
- data/: Eğitim verileri
- models/: Eğitilmiş modeller
- scripts/: Eğitim ve değerlendirme scriptleri

## Başlangıç
- `scripts/` klasörüne örnek eğitim scripti ekleyebilirsiniz.
- Hugging Face Dataset ve Model Hub kullanılabilir.

---

Hugging face arayüzü öğreniliyor