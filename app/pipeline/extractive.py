"""LexRank-сжатие длинных текстов перед abstractive-моделью."""
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.stemmers import Stemmer
from app.pipeline.preprocess import get_russian_stopwords


def lexrank_compress(text: str, target_sentences: int = 25) -> str:
    """Оставляет top-N самых "центральных" предложений в исходном порядке."""
    parser = PlaintextParser.from_string(text, Tokenizer("russian"))
    stemmer = Stemmer("russian")
    summarizer = LexRankSummarizer(stemmer)
    summarizer.stop_words = get_russian_stopwords()

    selected = summarizer(parser.document, target_sentences)
    # Sumy возвращает предложения в порядке "важности" — восстанавливаем исходный порядок.
    selected_strs = {str(s) for s in selected}
    ordered = [str(s) for s in parser.document.sentences if str(s) in selected_strs]
    return " ".join(ordered)