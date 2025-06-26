from transformers import pipeline

def main():
    ceviri = pipeline("translation", model="Helsinki-NLP/opus-mt-tr-en")
    metin = "Adalet mülkün temelidir."
    sonuc = ceviri(metin)
    print("Türkçe:", metin)
    print("İngilizce:", sonuc[0]['translation_text'])

if __name__ == "__main__":
    main()