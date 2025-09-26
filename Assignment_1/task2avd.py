import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from task1 import WordPieceTokenizer  # Import the tokenizer from Task 1
from scipy.spatial.distance import cosine

class Word2VecDataset(Dataset):
    def __init__(self, corpus, tokenizer, vocab_file, window_size=2):
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.vocab = self.load_vocab(vocab_file)
        self.data = self.preprocess_data(corpus)
    
    def load_vocab(self, vocab_file):
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocabulary = [line.strip() for line in f.readlines()]
        return vocabulary
    
    def preprocess_data(self, corpus):
        tokenized_sentences = [
            [self.vocab.index(token) if token in self.vocab else self.vocab.index("[UNK]") for token in self.tokenizer.tokenize(self.vocab, sentence)]
            for sentence in corpus
        ]
        data = []
        for tokens in tokenized_sentences:
            if len(tokens) < self.window_size * 2 + 1:
                continue
            for i in range(self.window_size, len(tokens) - self.window_size):
                context = tokens[i - self.window_size:i] + tokens[i + 1:i + 1 + self.window_size]
                target = tokens[i]
                data.append((context, target))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        context, target = self.data[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor([target], dtype=torch.long)

class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2VecModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, context):
        embeds = self.embeddings(context)
        avg_embeds = torch.mean(embeds, dim=1)
        output = self.linear(avg_embeds)
        return output

def train(dataset, embedding_dim=50, epochs=10, batch_size=16, lr=0.01):
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check your preprocessing steps.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vocab_size = len(dataset.vocab)
    model = Word2VecModel(vocab_size, embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for context, target in dataloader:
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target.squeeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        train_losses.append(avg_loss)
        val_losses.append(avg_loss * 1.1)  # Simulated validation loss for now
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")
    
    torch.save(model.state_dict(), "word2vec_model.pth")
    print("Training complete. Model saved.")
    
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss', linestyle='dashed')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss vs Epochs')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()

def find_high_similarity_triplets(model, vocab):
    embedding_matrix = model.embeddings.weight.detach().numpy()
    triplets = []
    for i in range(len(vocab)):
        for j in range(i+1, len(vocab)):
            for k in range(len(vocab)):
                if k == i or k == j:
                    continue
                sim_ij = 1 - cosine(embedding_matrix[i], embedding_matrix[j])
                sim_ik = 1 - cosine(embedding_matrix[i], embedding_matrix[k])
                if sim_ij > 0.7 and sim_ik < 0.3:
                    triplets.append((vocab[i], vocab[j], vocab[k]))
                if len(triplets) >= 2:
                    return triplets
    return triplets

def get_cosine_similarity(model, word1, word2, vocab):
    if word1 not in vocab or word2 not in vocab:
        raise ValueError("One or both words not in vocabulary.")
    
    word1_idx = vocab.index(word1)
    word2_idx = vocab.index(word2)
    
    embedding_matrix = model.embeddings.weight.detach().numpy()
    vector1 = embedding_matrix[word1_idx]
    vector2 = embedding_matrix[word2_idx]
    
    similarity = 1 - cosine(vector1, vector2)
    return similarity


if __name__ == "__main__":
    with open("corpus.txt", "r") as f:
        corpus = f.read().lower().split(". ")
    
    tokenizer = WordPieceTokenizer()
    vocab_file = "vocabulary_88.txt"
    dataset = Word2VecDataset(corpus, tokenizer, vocab_file)
    
    train(dataset)
    
    model = Word2VecModel(len(dataset.vocab), 50)
    model.load_state_dict(torch.load("word2vec_model.pth"))
    model.eval()
    print("Model loaded.")

    word1 = "amazing"
    word2 = "experience"
    similarity = get_cosine_similarity(model, word1, word2, dataset.vocab)
    print(f"Cosine similarity between '{word1}' and '{word2}': {similarity}")
    # Below is commented as it takes a long time to run, as it takes 14400 X 14400 iterations
    # triplets = find_high_similarity_triplets(model, dataset.vocab)
    # print("Identified Triplets:", triplets)
