import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import argparse
from src.transformer import Transformer, TransformerConfig, Vocabulary, QADataset, tokenizer

def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for src, trg in dataloader:
            src, trg = src.to(device), trg.to(device)

            src_mask = model.encoder.create_src_mask(src)
            trg_mask = model.decoder.create_trg_mask(trg[:, :-1])

            trg_input = trg[:, :-1]
            trg_output = trg[:, 1:].contiguous().view(-1)

            optimizer.zero_grad()
            output, _, _, _ = model(src, trg_input, src_mask, trg_mask)
            output = output.view(-1, output.size(-1))

            loss = criterion(output, trg_output)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    return model

def main():
    parser = argparse.ArgumentParser(description="Train a Transformer model for question answering.")
    parser.add_argument("--data_path", type=str, default="data.json", help="Path to the dataset JSON file.")
    parser.add_argument("--model_path", type=str, default="model.pth", help="Path to save/load model weights.")
    parser.add_argument("--vocab_path", type=str, default="vocabulary.json", help="Path to save/load vocabulary.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    args = parser.parse_args()

    # Load data
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = TransformerConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(args.vocab_path):
        with open(args.vocab_path, "r", encoding="utf-8") as f:
            word2idx = json.load(f)
        vocab = Vocabulary(min_freq=1)
        vocab.word2idx = word2idx
        vocab.idx2word = {v: k for k, v in word2idx.items()}
        vocab.vocab_size = len(word2idx)
        print("Loaded existing vocabulary.")
    else:
        all_texts = [pair[0] + pair[1] for pair in [(tokenizer(item['input']), 
                                                      tokenizer(item['output'])) for item in data]]
        vocab = Vocabulary(min_freq=1)
        vocab.build_vocab(all_texts)
        with open(args.vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab.word2idx, f, ensure_ascii=False)
        print("Built and saved new vocabulary.")

    config.src_vocab_size = vocab.vocab_size
    config.trg_vocab_size = vocab.vocab_size

    # Initialize
    model = Transformer(config).to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Loaded existing model weights.")
    else:
        print("No saved model found, starting with fresh weights.")

    
    dataset = QADataset(data, vocab, config.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    
    model = train_model(model, dataloader, criterion, optimizer, args.epochs, device)

    # Save model and vocabulary
    torch.save(model.state_dict(), args.model_path)
    with open(args.vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab.word2idx, f, ensure_ascii=False)
    print(f"Model saved to {args.model_path}")
    print(f"Vocabulary saved to {args.vocab_path}")

if __name__ == "__main__":
    main()
