from transformers import pipeline

def main():
    analiz = pipeline("sentiment-analysis")
    metin = "Bu ülkeyi çok seviyorum, her şey daha güzel olacak."
    sonuc = analiz(metin)
    print("Metin:", metin)
    print("Duygu:", sonuc[0]['label'], "Güven:", round(sonuc[0]['score'] * 100, 2), "%")

if __name__ == "__main__":
    main()