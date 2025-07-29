import torch
import torch.nn as nn
from tokenizer_utils import TokenizerEnVi
from model import build_gpt_model
import torch.nn.functional as F
# --- Cấu hình ---
SEQ_LEN = 256
MODEL_PATH = "saved/best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load tokenizer ---
tokenizer = TokenizerEnVi(language="vi")
tokenizer.special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]

# Tải vocab (nếu bạn đã lưu), ví dụ:
import pickle
with open("saved/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

pad_id = tokenizer.word2idx["<pad>"]

# --- Load model ---
model = build_gpt_model(
    vocab_size=len(tokenizer.word2idx),
    seq_len=SEQ_LEN,
    d_model=128,
    N=6,
    h=8,
    dropout=0.1,
    d_ff=512
).to(DEVICE)

model.load_state_dict(torch.load("saved/best_model.pth", map_location=DEVICE))
model.eval()




def generate_response(model, tokenizer, prompt,
                      device,
                      seq_len=256,
                      max_new_tokens=50,
                      temperature=0.7,
                      top_k=30):
    model.eval()
    with torch.no_grad():
        # 1. Chuẩn bị prompt
        text = f"<sos> USER {prompt} AI"
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        # 2. Truncate
        if len(input_ids) > seq_len - 1:
            input_ids = input_ids[: seq_len - 1]
        input_ids = input_ids[:]  # copy
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, L)

        for _ in range(max_new_tokens):
            L = input_tensor.size(1)
            attn_mask = torch.tril(torch.ones((1, 1, L, L), dtype=torch.bool, device=device))

            # 3. Forward
            logits = model(input_tensor, attn_mask)   # (1, L, V)
            next_logits = logits[0, -1, :]            # (V,)

            # 4. Temperature
            next_logits = next_logits / temperature

            # 5. Top-k filtering on logits
            if top_k > 0:
                values, indices = torch.topk(next_logits, top_k)
                mask = torch.full_like(next_logits, -float('Inf'))
                mask[indices] = next_logits[indices]
                next_logits = mask

            # 6. Softmax → probs
            probs = F.softmax(next_logits, dim=-1)
            # colon_id = tokenizer.word2idx.get(":")
            # if colon_id is not None:
            #     probs[colon_id] = 0
            #     probs = probs / probs.sum()
            next_id = torch.multinomial(probs, 1).item()

            # 8. Nếu gặp <eos> dừng
            if tokenizer.idx2word.get(next_id) == "<eos>":
                break

            # 9. Nối
            input_tensor = torch.cat(
                [input_tensor, torch.tensor([[next_id]], device=device)], dim=1
            )
            # 10. Truncate nếu vượt seq_len
            if input_tensor.size(1) > seq_len:
                input_tensor = input_tensor[:, -seq_len:]

        # 11. Decode toàn bộ chuỗi
        output_ids = input_tensor.squeeze().tolist()
        text_out = tokenizer.decode(output_ids, skip_special_tokens=True)

        # 12. Lấy phần sau "bot:"
        if "AI" in text_out:
            return text_out.split("AI", 1)[1].strip()
        return text_out.strip()



# --- Vòng lặp chat ---
if __name__ == "__main__":
    print("ChatGPT mini - type 'exit' to quit")
    while True:
        user_input = input("Bạn: ")
        if user_input.strip().lower() == "exit":
            break
        response = generate_response(model, tokenizer, user_input, DEVICE)
        print("AI:", response)
