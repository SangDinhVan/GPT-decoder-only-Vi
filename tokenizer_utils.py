import re
from collections import Counter
from pyvi import ViTokenizer



class TokenizerEnVi:
    def __init__(self, language='vi', min_freq=2):
        self.language = language
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]

        self.token_map = {tok: f"__{tok.strip('<>:').upper()}__" for tok in self.special_tokens}

    def tokenize(self, text):
        # 1. Thay thế special token bằng placeholder
        for orig, placeholder in self.token_map.items():
            text = text.replace(orig, f" {placeholder} ")

        # 2. Dùng PyVi tokenize toàn bộ câu
        tokens = ViTokenizer.tokenize(text).split()

        # 3. Đổi lại placeholder thành special token gốc
        final_tokens = []
        for tok in tokens:
            if tok in self.token_map.values():
                # map ngược lại
                orig = [k for k, v in self.token_map.items() if v == tok][0]
                final_tokens.append(orig)
            else:
                final_tokens.append(tok)

        return final_tokens

    def build_vocab(self, sentences):
        # Bước 1: Tokenize toàn bộ corpus
        tokenized_sentences = [self.tokenize(sentence) for sentence in sentences]
        counter = Counter(token for sent in tokenized_sentences for token in sent)

        # Bước 2: Tạo set từ valid tokens
        valid_tokens = [word for word, freq in counter.items()
                        if freq >= self.min_freq and self.is_valid_token(word)]

        # Bước 3: Gộp special tokens (duy nhất, không trùng)
        vocab = []
        seen = set()
        for token in self.special_tokens:
            if token not in seen:
                vocab.append(token)
                seen.add(token)

        for word in valid_tokens:
            if word not in seen:
                vocab.append(word)
                seen.add(word)

        # Bước 4: Gán chỉ số liên tục từ 0
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def encode(self, sentence, add_special_tokens=True, max_len=None):
        # Cho phép truyền list token hoặc chuỗi
        if isinstance(sentence, str):
            tokens = self.tokenize(sentence)
        else:
            tokens = sentence  # đã là list token

        if add_special_tokens:
            tokens = ["<sos>"] + tokens + ["<eos>"]

        if max_len:
            tokens = tokens[:max_len]

        ids = [self.word2idx.get(token, self.word2idx["<unk>"]) for token in tokens]
        return ids

    def decode(self, ids, skip_special_tokens=True):
        tokens = []
        for idx in ids:
            token = self.idx2word.get(idx, "<unk>")
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def vocab_size(self):
        return len(self.word2idx)

    def pad_sequence(self, ids, max_len):
        if len(ids) > max_len:
            return ids[:max_len]
        pad_id = self.word2idx.get("<pad>", 0)
        return ids + [pad_id] * (max_len - len(ids))

    def is_valid_token(self, token):
        # Luôn giữ special token
        if token in self.special_tokens:
            return True
        # Loại token có ký tự lạ
        if any(c in token for c in ['<', '>', '[', ']', '(', ')', '{', '}', '=', '/', '\\', '"']):
            return False
        # Loại token toàn số
        if token.isdigit():
            return False
        # Loại token quá ngắn hoặc quá dài
        if len(token) < 1 or len(token) > 30:
            return False
        return True
