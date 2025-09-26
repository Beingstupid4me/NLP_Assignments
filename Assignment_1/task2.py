import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from collections import defaultdict
from task1 import WordPieceTokenizer  # Import the tokenizer from Task 1

class Word2VecDataset(Dataset):
    def __init__(self, corpus, tokenizer, vocab_file, window_size=2):
        """
        Custom dataset for Word2Vec CBOW training.
        :param corpus: List of sentences (raw text)
        :param tokenizer: Instance of WordPieceTokenizer
        :param vocab_file: Path to the vocabulary file
        :param window_size: Number of words before and after the target word
        """
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.vocab = self.load_vocab(vocab_file)
        self.data = self.preprocess_data(corpus)
    
    def load_vocab(self, vocab_file):
        """Loads vocabulary from a file."""
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocabulary = [line.strip() for line in f.readlines()]
        return vocabulary
    
    def preprocess_data(self, corpus):
        """Tokenizes the text corpus and prepares CBOW training data."""
        tokenized_sentences = [
            [self.vocab.index(token) if token in self.vocab else self.vocab.index("[UNK]") for token in self.tokenizer.tokenize(self.vocab, sentence)]
            for sentence in corpus
        ]
        data = []
        for tokens in tokenized_sentences:
            if len(tokens) < self.window_size * 2 + 1:
                continue  # Skip sentences that are too short
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
        """
        Implements CBOW Word2Vec model.
        :param vocab_size: Total number of unique words
        :param embedding_dim: Size of embedding vectors
        """
        super(Word2VecModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, context):
        """Forward pass to predict target word from context."""
        embeds = self.embeddings(context)
        avg_embeds = torch.mean(embeds, dim=1)
        output = self.linear(avg_embeds)
        return output

def train(dataset, embedding_dim=50, epochs=10, batch_size=16, lr=0.01):
    """Handles training of the Word2Vec model."""
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check your preprocessing steps.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vocab_size = len(dataset.vocab)
    model = Word2VecModel(vocab_size, embedding_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for context, target in dataloader:
            optimizer.zero_grad()
            output = model(context)
            loss = criterion(output, target.squeeze(1))  # Ensure correct shape
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader)}")
    
    torch.save(model.state_dict(), "word2vec_model.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    # Load corpus from a sample file (Modify as needed)
    with open("corpus.txt", "r") as f:
        corpus = f.read().lower().split(". ")  # Simple sentence splitting
    
    tokenizer = WordPieceTokenizer()
    vocab_file = "vocabulary_88.txt"  # Path to the predefined vocabulary
    dataset = Word2VecDataset(corpus, tokenizer, vocab_file)
    
    train(dataset)