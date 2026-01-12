import zeyrek
from .word_tokenizer import TurkishWordTokenizer


class TurkishWordZeyrekTokenizer(TurkishWordTokenizer):
    """
    Türkçe kelimeler için:
    - Önce standart word tokenization yapar
    - Ardından Zeyrek ile morfolojik çözümleme (lemma) uygular
    - Performans için kelime bazlı cache kullanır
    """

    def __init__(self):
        # Üst sınıftaki (TurkishWordTokenizer) ayarları başlat
        super().__init__()

        # Zeyrek morfolojik analiz nesnesi
        self.analyzer = zeyrek.MorphAnalyzer()

        # Aynı kelimeyi tekrar analiz etmemek için cache
        # Anahtar: kelime, Değer: lemma
        self.cache = {}
    def tokenize(self, text):
        tokens = super().tokenize(text)
        lemmas = []

        for token in tokens:
            # Cache kontrolü
            if token in self.cache:
                lemmas.append(self.cache[token])
                continue

            # Zeyrek morfolojik analiz
            analyses = self.analyzer.analyze(token)

            # Zeyrek çıktısı: List[List[Analysis]]
            if analyses and analyses[0]:
                lemma = analyses[0][0].lemma
            else:
                lemma = token  # fallback

            # Cache'e yaz
            self.cache[token] = lemma
            lemmas.append(lemma)

        return lemmas
