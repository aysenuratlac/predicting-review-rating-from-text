"""
Turkish Word Tokenizer
NLTK word_tokenize kullanarak
"""

import re
import nltk
from nltk.tokenize import word_tokenize

class TurkishWordTokenizer:
    """
    Türkçe kelime tokenizer - sklearn ile uyumlu
    
    NLTK word_tokenize kullanır, Türkçe dil desteği ile.
    
    Usage:
        tokenizer = TurkishWordTokenizer()
        words = tokenizer.tokenize("Harika bir restoran!")
        # ['harika', 'bir', 'restoran']
        
        # sklearn ile kullanım:
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize)
    """
    
    def __init__(self, lowercase=True, remove_punctuation=True, remove_stopwords=False):
        """
        Parameters:
        -----------
        lowercase : bool, default=True
            Metni küçük harfe çevir
        remove_punctuation : bool, default=True
            Noktalama işaretlerini kaldır
        remove_stopwords : bool, default=False
            Türkçe stopword'leri kaldır
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        
        # NLTK punkt tokenizer kontrol
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            print("⚠️  NLTK punkt tokenizer indiriliyor...")
            nltk.download('punkt', quiet=True)
        
        # Türkçe stopwords
        self.turkish_stopwords = set([
            've', 'veya', 'ama', 'fakat', 'ancak', 'lakin',
            'için', 'ile', 'da', 'de', 'ki', 'mi', 'mı', 'mu', 'mü',
            'bir', 'bu', 'şu', 'o', 'her', 'bazı', 'çok', 'az',
            'ne', 'nasıl', 'neden', 'niçin', 'nerede', 'kim', 'ben', 'sen',
            'biz', 'siz', 'onlar', 'şey', 'gibi', 'kadar', 'daha',
            'en', 'pek', 'oldukça', 'son', 'ilk', 'var', 'yok', 'olan'
        ])
        
        print(f"✅ TurkishWordTokenizer hazır! (NLTK)")
    
    def tokenize(self, text):
        """
        Metni kelimelere ayır
        
        Parameters:
        -----------
        text : str
            Tokenize edilecek metin
            
        Returns:
        --------
        list of str
            Kelimeler listesi
        """
        if not text or not isinstance(text, str):
            return []
        
        # NLTK ile tokenize (Türkçe dil desteği)
        tokens = word_tokenize(text, language='turkish')
        
        # Lowercase
        if self.lowercase:
            tokens = [t.lower() for t in tokens]
        
        # Noktalama kaldır
        if self.remove_punctuation:
            # Sadece harf/rakam içeren token'ları tut
            tokens = [t for t in tokens if re.search(r'\w', t)]
        
        # Stopword kaldır
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.turkish_stopwords]
        
        return tokens
    
    def __call__(self, text):
        """
        sklearn uyumluluğu için
        """
        return self.tokenize(text)


def turkish_word_tokenizer(text):
    """
    Basit fonksiyon wrapper (sklearn için)
    """
    # Global obje (ilk kullanımda oluştur)
    if not hasattr(turkish_word_tokenizer, '_tokenizer'):
        turkish_word_tokenizer._tokenizer = TurkishWordTokenizer()
    
    return turkish_word_tokenizer._tokenizer.tokenize(text)


# Test
if __name__ == "__main__":
    print("🧪 Kelime Tokenizer Test\n")
    
    # Tokenizer oluştur
    tokenizer = TurkishWordTokenizer(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=False
    )
    
    # Test metinleri
    test_texts = [
        "Harika bir restoran! Çok beğendim.",
        "Güzel mekan ama yemekler soğuktu.",
        "Istanbul'un en iyi yerlerinden biri.",
        "5/5 puan veriyorum, mükemmel hizmet."
    ]
    
    print("Test metinleri:\n")
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        print(f"  Metin: {text}")
        print(f"  Tokenlar: {tokens}")
        print(f"  Toplam: {len(tokens)} token\n")
    
    # Stopword ile test
    print("\n" + "="*50)
    print("Stopword kaldırma ile test:\n")
    tokenizer_no_stop = TurkishWordTokenizer(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True
    )
    
    sample = "Bu bir harika restoran ve çok güzel"
    print(f"Metin: {sample}")
    print(f"Normal: {tokenizer.tokenize(sample)}")
    print(f"Stopword'sız: {tokenizer_no_stop.tokenize(sample)}")
    
