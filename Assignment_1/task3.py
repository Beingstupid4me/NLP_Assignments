import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from task1 import WordPieceTokenizer
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import math

class WordEmbeddingDataset(Dataset):
    def __init__(self, file_path, context_window=3, tokenizer=None, word2vec_model=None):
        self.tokenizer = tokenizer if tokenizer else WordPieceTokenizer()  # Using custom tokenizer
        self.embeddings = word2vec_model.wv  # Extracting word embeddings from Word2Vec model
        self.context_window = context_window
        self.pairs = []
        self.prepare_data(file_path)

    def prepare_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            sentences = file.readlines()

        for sentence in sentences:
            tokens = self.tokenizer.breakingOfTokens(sentence.strip())
            for idx in range(len(tokens) - self.context_window):
                context_words = tokens[idx:idx + self.context_window]
                target_word = tokens[idx + self.context_window]

                if target_word in self.embeddings:
                    context_vector = torch.mean(torch.stack([
                        torch.tensor(self.embeddings[word], dtype=torch.float32) for word in context_words if word in self.embeddings
                    ]), dim=0)
                    target_idx = self.embeddings.key_to_index[target_word]  # Get target word index
                    self.pairs.append((context_vector, target_idx))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx][0], self.pairs[idx][1]  # Target is now an index

class SimpleLM1(nn.Module):
    def __init__(self, vector_size=100, vocab_size=None):
        super().__init__()
        self.fc = nn.Linear(vector_size, vocab_size)  # Output layer size is vocab_size
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        return x  # Don't apply softmax here; cross_entropy will handle it

class SimpleLM2(nn.Module):
    def __init__(self, vector_size=100, vocab_size=None):
        super().__init__()
        self.fc1 = nn.Linear(vector_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, vocab_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)  # Output logits

class SimpleLM3(nn.Module):
    def __init__(self, vector_size=100, vocab_size=None):
        super().__init__()
        self.fc1 = nn.Linear(vector_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, vocab_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # Output logits

def train_model(model, train_dataloader, val_dataloader, vocab_size, epochs=2, lr=0.001):
    loss_function = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for context, target in train_dataloader:
            optimizer.zero_grad()
            predictions = model(context)  # Logits, not softmax
            loss = loss_function(predictions, target)  # Target should be indices
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for context, target in val_dataloader:
                predictions = model(context)  # Logits
                loss = loss_function(predictions, target)
                total_val_loss += loss.item()

        train_losses.append(total_train_loss / len(train_dataloader))
        val_losses.append(total_val_loss / len(val_dataloader))

        print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]}, Val Loss: {val_losses[-1]}")

        # Compute Accuracy and Perplexity
        train_accuracy = calculate_accuracy(model, train_dataloader)
        val_accuracy = calculate_accuracy(model, val_dataloader)
        train_perplexity = calculate_perplexity(model, train_dataloader)
        val_perplexity = calculate_perplexity(model, val_dataloader)

        print(f"Train Accuracy: {train_accuracy:.2f}% | Val Accuracy: {val_accuracy:.2f}%")
        print(f"Train Perplexity: {train_perplexity:.2f} | Val Perplexity: {val_perplexity:.2f}")

    # Plotting the loss graph
    plt.plot(range(epochs), train_losses, label='Training Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss vs Epochs')
    plt.show()

def calculate_accuracy(model, dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for context, target in dataloader:
            predictions = model(context)
            _, predicted_idx = torch.max(predictions, dim=1)  # Max logit prediction
            correct += (predicted_idx == target).sum().item()
            total += target.size(0)
    accuracy = correct / total * 100
    return accuracy

def calculate_perplexity(model, dataloader):
    loss_function = nn.CrossEntropyLoss()
    total_loss = 0
    total_words = 0
    with torch.no_grad():
        for context, target in dataloader:
            predictions = model(context)
            loss = loss_function(predictions.view(-1, predictions.size(-1)), target.view(-1))
            total_loss += loss.item() * context.size(0)
            total_words += context.size(0)
    perplexity = math.exp(total_loss / total_words)
    return perplexity


def get_next_word_predictions(model, sentence, word2vec_model, top_k=3):
    tokenizer = WordPieceTokenizer()  # Custom tokenizer

    # Accessing the word2vec embeddings directly
    embeddings = word2vec_model.wv

    tokens = tokenizer.breakingOfTokens(sentence.strip())
    input_vector = torch.mean(torch.stack([
        torch.tensor(embeddings[word], dtype=torch.float32) for word in tokens if word in embeddings
    ]), dim=0)

    prediction_vector = model(input_vector.unsqueeze(0))

    # Convert numpy.ndarray to tensor for embeddings
    all_embeddings = torch.tensor(embeddings.vectors, dtype=torch.float32)
    similarity_scores = torch.matmul(prediction_vector, all_embeddings.T)

    top_indices = torch.argsort(similarity_scores, descending=True)[0][:top_k]

    predicted_words = [list(embeddings.index_to_key)[i] for i in top_indices]
    return predicted_words


def train_word2vec_model(corpus_file, vector_size=100, window=5):
    with open(corpus_file, 'r', encoding='utf-8') as file:
        corpus = file.readlines()

    tokenized_sentences = [WordPieceTokenizer().breakingOfTokens(sentence.strip()) for sentence in corpus]
    word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=vector_size, window=window, min_count=1, workers=4)
    return word2vec_model

def main():
    corpus_file = "corpus.txt"
    word2vec_model = train_word2vec_model(corpus_file)
    vocab_size = len(word2vec_model.wv)  # Get vocabulary size

    dataset = WordEmbeddingDataset(corpus_file, tokenizer=WordPieceTokenizer(), word2vec_model=word2vec_model)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Split dataset for training and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    for model_class in [SimpleLM1, SimpleLM2, SimpleLM3]:
        print(f"Training {model_class.__name__}")
        model = model_class(vocab_size=vocab_size)
        train_model(model, train_dataloader, val_dataloader, vocab_size)

    with open("sample_test.txt", 'r', encoding='utf-8') as f:
        test_sentences = f.readlines()

    for sentence_data in test_sentences:
        sentence = sentence_data.strip()
        print(f"Input: {sentence} \nPredicted: {get_next_word_predictions(model, sentence)}\n")

if __name__ == "__main__":
    main()
