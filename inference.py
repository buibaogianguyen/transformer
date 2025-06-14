import torch
import json
import os
import argparse
from src.transformer import Transformer, TransformerConfig, Vocabulary, tokenizer

def inference(model, prompt, vocab, config, device, max_len=256):
    model.eval()
    with torch.no_grad():
        tokens = tokenizer(prompt)
        src_indices = vocab.tokenize_to_indices(tokens, max_len)
        src = torch.tensor([src_indices], dtype=torch.long).to(device)

        src_mask = model.encoder.create_src_mask(src).to(device)
        enc_output, _ = model.encoder(src, src_mask)

        trg_indices = [vocab.word2idx["<GO>"]]
        trg = torch.tensor([trg_indices], dtype=torch.long).to(device)

        output_words = []
        for _ in range(max_len):
            trg_mask = model.decoder.create_trg_mask(trg).to(device)
            output, _, _ = model.decoder(trg, enc_output, src_mask, trg_mask)
            output = output[:, -1, :]
            pred_token = output.argmax(dim=-1).item()

            trg_indices.append(pred_token)
            trg = torch.tensor([trg_indices], dtype=torch.long).to(device)

            pred_word = vocab.idx2word.get(pred_token, "<UNK>")
            output_words.append(pred_word)

            if pred_token == vocab.word2idx["<EOS>"]:
                break

        output_text = " ".join([word for word in output_words if word not in ["<GO>", "<EOS>"]])
        return output_text

def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained Transformer model.")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for the model.")
    parser.add_argument("--model_path", type=str, default="model.pth", help="Path to the trained model weights.")
    parser.add_argument("--vocab_path", type=str, default="vocabulary.json", help="Path to the vocabulary file.")
    parser.add_argument("--max_len", type=int, default=256, help="Maximum sequence length for inference.")
    args = parser.parse_args()

    # Load vocab
    with open(args.vocab_path, "r", encoding="utf-8") as f:
        word2idx = json.load(f)
    vocab = Vocabulary(min_freq=1)
    vocab.word2idx = word2idx
    vocab.idx2word = {v: k for k, v in word2idx.items()}
    vocab.vocab_size = len(word2idx)

    # Initialize
    config = TransformerConfig()
    config.src_vocab_size = vocab.vocab_size
    config.trg_vocab_size = vocab.vocab_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(config).to(device)

    # Load model weights
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Loaded model weights.")
    else:
        raise FileNotFoundError(f"Model file {args.model_path} not found.")

    # Inference
    response = inference(model, args.prompt, vocab, config, device, max_len=args.max_len)
    print(f"Prompt: {args.prompt}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()