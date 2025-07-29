from tokenizer_utils import TokenizerEnVi
from sklearn.model_selection import train_test_split

with open("data/data.txt", encoding="utf-8") as f:
    lines1 = [line.strip() for line in f if "<sos>" in line and "<eos>" in line]

with open("data/data2.txt", encoding="utf-8") as f:
    lines2 = [line.strip() for line in f if "<sos>" in line and "<eos>" in line]

lines = lines1 + lines2

train_lines, test_lines = train_test_split(lines, test_size=0.2, random_state = 42)

tokenizer = TokenizerEnVi()
tokenizer.build_vocab(train_lines)

for i,j in tokenizer.word2idx.items():
    print(i,j)