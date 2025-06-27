from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)
ceviri = pipeline("translation", model="Helsinki-NLP/opus-mt-tr-en")

@app.route("/", methods=["GET", "POST"])
def index():
    sonuc = ""
    if request.method == "POST":
        metin = request.form["metin"]
        ceviri_sonucu = ceviri(metin)
        sonuc = ceviri_sonucu[0]['translation_text']
    return render_template("index.html", sonuc=sonuc)

if __name__ == "__main__":
    app.run(debug=True)