import torch
from torch.utils.data import Dataset

class GPTChatDataset(Dataset):
    def __init__(self, lines, tokenizer, seq_len):
        self.lines = lines
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_id = tokenizer.word2idx["<pad>"]
        self.bot_token_id = tokenizer.word2idx.get("<bot>", None)
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        text = self.lines[idx].strip()

        input_ids = self.tokenizer.encode(text, add_special_tokens=False)

        input_ids = self.pad_sequence(input_ids)

        # Tạo attention mask (causal mask)
        attn_mask = self.generate_causal_mask(self.seq_len)

        # Tạo label giống input, nhưng mask phần prompt (trước "bot:")
        labels = input_ids.copy()
        labels = [tok if tok != self.pad_id else -100 for tok in input_ids]

        if self.bot_token_id is not None:
            try:
                bot_index = input_ids.index(self.bot_token_id)
                for i in range(bot_index + 1):
                    labels[i] = -100
            except ValueError:
                labels = [-100] * len(input_ids)
        else:
            labels = [-100] * len(input_ids)

        return {
            "decoder_input": torch.tensor(input_ids),
            "decoder_mask": attn_mask,
            "label": torch.tensor(labels)
        }

    def pad_sequence(self, ids):
        if len(ids) > self.seq_len:
            return ids[:self.seq_len]
        return ids + [self.pad_id] * (self.seq_len - len(ids))

    def generate_causal_mask(self, seq_len):
        return torch.tril(torch.ones((1, seq_len, seq_len), dtype=torch.bool))
