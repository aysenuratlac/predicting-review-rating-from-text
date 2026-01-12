"""
Turkish Syllable Tokenizer
Uses turkishnlp library for syllabication
"""

from turkishnlp import detector
import re

class TurkishSyllableTokenizer:
    """
    Türkçe hece tokenizer - sklearn ile uyumlu
    
    turkishnlp kütüphanesini kullanarak metni hecelere ayırır.
    
    Usage:
        tokenizer = TurkishSyllableTokenizer()
        syllables = tokenizer.tokenize("Harika bir restoran")
        # ['ha', 'ri', 'ka', 'bir', 'res', 'to', 'ran']
        
        # sklearn ile kullanım:
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize)
    """
    
    def __init__(self, lowercase=True, remove_punctuation=True):
        """
        Parameters:
        -----------
        lowercase : bool, default=True
            Metni küçük harfe çevir
        remove_punctuation : bool, default=True
            Noktalama işaretlerini kaldır
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        
        # turkishnlp objesi oluştur
        self.nlp = detector.TurkishNLP()
        
        # Veri setlerini oluştur (ilk kulanımda gerekli)
        try:
            self.nlp.create_word_set()
        except:
            pass  # Zaten oluşturulmuşsa sorun yok
        
        print("✅ TurkishSyllableTokenizer hazır!")
    
    def tokenize(self, text):
        """
        Metni hecelere ayır
        
        Parameters:
        -----------
        text : str
            Tokenize edilecek metin
            
        Returns:
        --------
        list of str
            Heceler listesi
        """
        if not text or not isinstance(text, str):
            return []
        
        # Preprocessing
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            # Noktalama işaretlerini kaldır
            text = re.sub(r'[^\w\s]', '', text)
        
        # Boşluklara göre kelimelere ayır
        words = text.split()
        
        # Her kelimeyi hecelere ayır
        all_syllables = []
        for word in words:
            if word.strip():  # Boş değilse
                try:
                    syllables = self.nlp.syllabicate(word)
                    all_syllables.extend(syllables)
                except Exception as e:
                    # Hata olursa kelimeyi olduğu gibi ekle
                    all_syllables.append(word)
        
        return all_syllables
    
    def __call__(self, text):
        """
        sklearn uyumluluğu için
        """
        return self.tokenize(text)


def turkish_syllable_tokenizer(text):
    """
    Basit fonksiyon wrapper (sklearn için)
    
    NOT: Her çağrıda yeni obje oluşturmaz, bu yüzden daha verimli.
    Ama ilk kullanımda obje oluşturma maliyeti var.
    """
    # Global obje (ilk kullanımda oluştur)
    if not hasattr(turkish_syllable_tokenizer, '_tokenizer'):
        turkish_syllable_tokenizer._tokenizer = TurkishSyllableTokenizer()
    
    return turkish_syllable_tokenizer._tokenizer.tokenize(text)


# Test
if __name__ == "__main__":
    print("🧪 Hece Tokenizer Test\n")
    
    # Tokenizer oluştur
    tokenizer = TurkishSyllableTokenizer()
    
    # Test metinleri
    test_texts = [
        "Harika bir restoran!",
        "Çok güzel ve lezzetli yemekler.",
        "Istanbul'un en iyi mekanı.",
        "Berbat bir deneyim."
    ]
    
    print("Test metinleri:\n")
    for text in test_texts:
        syllables = tokenizer.tokenize(text)
        print(f"  Metin: {text}")
        print(f"  Heceler: {syllables}")
        print(f"  Toplam: {len(syllables)} hece\n")
    
