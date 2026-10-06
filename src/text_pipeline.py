import re
import logging
import pandas as pd
from typing import Union
from sklearn.base import BaseEstimator, TransformerMixin

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FINTECH_ACRONYMS = {
    "upi": "unified payments interface",
    "kyc": "know your customer",
    "neft": "national electronic funds transfer",
    "rtgs": "real time gross settlement",
    "imps": "immediate payment service",
    "atm": "automated teller machine",
    "pos": "point of sale",
    "nfc": "near field communication",
    "qr": "quick response",
    "api": "application programming interface",
    "sdk": "software development kit",
    "ai": "artificial intelligence",
    "ml": "machine learning",
    "nlp": "natural language processing",
    "ocr": "optical character recognition",
    "kyb": "know your business",
    "aml": "anti money laundering",
    "cft": "combating financing of terrorism",
    "psp": "payment service provider",
    "pg": "payment gateway",
    "bnpl": "buy now pay later",
    "emi": "equated monthly installment",
    "sip": "systematic investment plan",
    "mf": "mutual fund",
    "demat": "dematerialised account",
    "repo": "repurchase agreement",
    "reverse repo": "reverse repurchase agreement",
    "crr": "cash reserve ratio",
    "slr": "statutory liquidity ratio",
    "msme": "micro small and medium enterprises",
    "gst": "goods and services tax",
    "pan": "permanent account number",
    "aadhaar": "aadhaar",
    "digilocker": "digilocker",
    "ekyc": "electronic know your customer",
    "vkyc": "video know your customer",
    "ckyc": "central know your customer",
}

EMOJI_MAP = {
    "😡": " angry ",
    "😠": " angry ",
    "🤬": " angry ",
    "😤": " frustrated ",
    "😭": " very sad ",
    "😢": " sad ",
    "☹️": " sad ",
    "😞": " disappointed ",
    "😔": " disappointed ",
    "😕": " confused ",
    "🙁": " sad ",
    "😐": " neutral ",
    "😑": " neutral ",
    "😶": " speechless ",
    "🙂": " slightly happy ",
    "😊": " happy ",
    "😀": " very happy ",
    "😁": " very happy ",
    "😄": " very happy ",
    "😃": " very happy ",
    "🥰": " love ",
    "😍": " love ",
    "❤️": " love ",
    "💖": " love ",
    "💕": " love ",
    "👍": " thumbs up ",
    "👎": " thumbs down ",
    "👌": " ok ",
    "🙏": " thank you ",
    "🤝": " handshake ",
    "💪": " strong ",
    "🔥": " fire ",
    "✨": " sparkles ",
    "💯": " hundred percent ",
    "🎉": " celebration ",
    "😱": " shocked ",
    "😨": " scared ",
    "😰": " anxious ",
    "😰": " anxious ",
    "🤯": " mind blown ",
    "😴": " sleepy ",
    "🤒": " sick ",
    "🤕": " hurt ",
    "💸": " money ",
    "💰": " money ",
    "💵": " money ",
    "💳": " credit card ",
    "🏦": " bank ",
    "🏧": " atm ",
    "📱": " mobile ",
    "💻": " laptop ",
    "🌐": " internet ",
    "⚡": " fast ",
    "⏱️": " time ",
    "⏰": " alarm ",
    "🔔": " notification ",
    "📩": " message ",
    "📧": " email ",
    "🚫": " banned ",
    "❌": " cancel ",
    "✅": " success ",
    "⚠️": " warning ",
    "ℹ️": " info ",
    "❓": " question ",
    "❗": " important ",
    "‼️": " very important ",
}

URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
EMAIL_PATTERN = re.compile(r'\S+@\S+\.\S+')
PHONE_PATTERN = re.compile(r'(\+91[\-\s]?|0)?[6-9]\d{9}')
MENTION_PATTERN = re.compile(r'@\w+')
HASHTAG_PATTERN = re.compile(r'#\w+')
MULTI_SPACE_PATTERN = re.compile(r'\s+')
SPECIAL_CHAR_PATTERN = re.compile(r'[^\w\s\.\,\!\?\:\;\-\'\"]')


class HinglishTextCleaner(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer for cleaning Hinglish FinTech reviews.
    Expands acronyms, translates emojis, removes PII and noise.
    """

    def __init__(
        self,
        lowercase: bool = True,
        expand_acronyms: bool = True,
        translate_emojis: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_phones: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,
        remove_special_chars: bool = True,
    ):
        self.lowercase = lowercase
        self.expand_acronyms = expand_acronyms
        self.translate_emojis = translate_emojis
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_phones = remove_phones
        self.remove_mentions = remove_mentions
        self.remove_hashtags = remove_hashtags
        self.remove_special_chars = remove_special_chars

        self._acronym_pattern = re.compile(
            r'\b(' + '|'.join(re.escape(k) for k in FINTECH_ACRONYMS.keys()) + r')\b',
            flags=re.IGNORECASE
        )
        self._emoji_pattern = re.compile('|'.join(re.escape(k) for k in EMOJI_MAP.keys()))

    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None) -> 'HinglishTextCleaner':
        logging.info("HinglishTextCleaner: fit() called (no-op, stateless transformer).")
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError("DataFrame input must have exactly one column.")
            X = X.iloc[:, 0]

        if not isinstance(X, pd.Series):
            raise TypeError(f"Expected pd.Series or single-column pd.DataFrame, got {type(X)}")

        logging.info(f"HinglishTextCleaner: Transforming {len(X)} records...")
        cleaned = X.apply(self._clean_text)
        logging.info("HinglishTextCleaner: Transformation complete.")
        return cleaned

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        if self.lowercase:
            text = text.lower()

        if self.remove_urls:
            text = URL_PATTERN.sub(' ', text)

        if self.remove_emails:
            text = EMAIL_PATTERN.sub(' ', text)

        if self.remove_phones:
            text = PHONE_PATTERN.sub(' ', text)

        if self.remove_mentions:
            text = MENTION_PATTERN.sub(' ', text)

        if self.remove_hashtags:
            text = HASHTAG_PATTERN.sub(' ', text)

        if self.translate_emojis:
            text = self._emoji_pattern.sub(lambda m: EMOJI_MAP.get(m.group(), ' '), text)

        if self.expand_acronyms:
            text = self._acronym_pattern.sub(
                lambda m: FINTECH_ACRONYMS.get(m.group().lower(), m.group()), text
            )

        if self.remove_special_chars:
            text = SPECIAL_CHAR_PATTERN.sub(' ', text)

        text = MULTI_SPACE_PATTERN.sub(' ', text).strip()
        return text