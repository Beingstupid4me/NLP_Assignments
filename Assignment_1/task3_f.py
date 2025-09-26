import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from task1 import WordPieceTokenizer  # Assumes Task 1 is available
from task2 import Word2VecModel      # For loading pretrained embeddings

# ----------------------------
# 1. NeuralLMDataset
# ----------------------------
class NeuralLMDataset(Dataset):
    def __init__(self, corpus, tokenizer, vocab_file, context_size=4):
        """
        Prepares dataset for next-word prediction.
        :param corpus: List of sentences (raw text).
        :param tokenizer: Instance of WordPieceTokenizer.
        :param vocab_file: Path to the vocabulary file.
        :param context_size: Number of tokens in context (i.e., tokens before the target).
        """
        self.tokenizer = tokenizer
        self.context_size = context_size
        self.vocab, self.token_to_idx = self.load_vocab(vocab_file)
        self.data = self.preprocess_data(corpus)

    def load_vocab(self, vocab_file):
        """Loads vocabulary from file and creates a token-to-index mapping."""
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocabulary = [line.strip() for line in f.readlines()]
        token_to_idx = {token: idx for idx, token in enumerate(vocabulary)}
        return vocabulary, token_to_idx

    def preprocess_data(self, corpus):
        data = []
        for sentence in corpus:
            tokens = self.tokenizer.tokenize(self.vocab, sentence)
            if len(tokens) < self.context_size + 1:
                continue
            for i in range(self.context_size, len(tokens)):
                context = tokens[i - self.context_size:i]
                target = tokens[i]
                data.append((context, target))
        print(f"Total training samples: {len(data)}")
        return data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, target = self.data[idx]
        # Convert tokens to indices
        context_indices = [self.token_to_idx.get(tok, self.token_to_idx.get("[UNK]")) for tok in context]
        target_index = self.token_to_idx.get(target, self.token_to_idx.get("[UNK]"))
        return torch.tensor(context_indices, dtype=torch.long), torch.tensor(target_index, dtype=torch.long)

# ----------------------------
# 2. Neural LM Architectures
# ----------------------------

# NeuralLM1: Baseline with a single hidden layer and ReLU activation.
class NeuralLM1(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim=128):
        super(NeuralLM1, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # If available, you could load pretrained embeddings from Task 2’s checkpoint here.
        self.fc1 = nn.Linear(embedding_dim * context_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context):
        embeds = self.embedding(context)  # [batch_size, context_size, embedding_dim]
        embeds = embeds.view(embeds.size(0), -1)  # Flatten context tokens
        out = self.fc1(embeds)
        out = self.relu(out)
        logits = self.fc2(out)
        return logits

# NeuralLM2: Deeper network with two hidden layers and dropout.
class NeuralLM2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim=256, dropout_p=0.3):
        super(NeuralLM2, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * context_size, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, vocab_size)

    def forward(self, context):
        embeds = self.embedding(context)
        embeds = embeds.view(embeds.size(0), -1)
        out = self.fc1(embeds)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        logits = self.fc3(out)
        return logits

# NeuralLM3: Uses LeakyReLU, an extra hidden layer, and a larger context window (if desired).
class NeuralLM3(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_dim=256, dropout_p=0.4):
        super(NeuralLM3, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * context_size, hidden_dim)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, context):
        embeds = self.embedding(context)
        embeds = embeds.view(embeds.size(0), -1)
        out = self.fc1(embeds)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.leaky_relu(out)
        logits = self.fc3(out)
        return logits

# ----------------------------
# 3. Training and Evaluation Functions
# ----------------------------
def compute_accuracy(logits, targets):
    """Computes accuracy given logits and target indices."""
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

def compute_perplexity(loss):
    """Computes perplexity from cross entropy loss."""
    return math.exp(loss)

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):
    """
    Trains a given model and returns training history.
    :return: dictionaries for train and validation losses, and accuracies.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        for contexts, targets in train_loader:
            contexts = contexts.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(contexts)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += compute_accuracy(logits, targets)
        avg_loss = epoch_loss / len(train_loader)
        avg_acc = epoch_acc / len(train_loader)
        train_losses.append(avg_loss)
        train_accuracies.append(avg_acc)

        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for contexts, targets in val_loader:
                contexts = contexts.to(device)
                targets = targets.to(device)
                logits = model(contexts)
                loss = criterion(logits, targets)
                val_loss += loss.item()
                val_acc += compute_accuracy(logits, targets)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)

        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Train Acc={avg_acc:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={avg_val_acc:.4f}")
    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accuracies,
        'val_acc': val_accuracies
    }

def plot_history(histories, model_names):
    """Plots training and validation loss for multiple models."""
    epochs = range(1, len(histories[0]['train_loss']) + 1)
    plt.figure(figsize=(12, 5))
    for i, history in enumerate(histories):
        plt.plot(epochs, history['train_loss'], label=f"{model_names[i]} Train Loss")
        plt.plot(epochs, history['val_loss'], label=f"{model_names[i]} Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.show()

# ----------------------------
# 4. Next-Token Prediction Pipeline
# ----------------------------
def predict_next_tokens(model, tokenizer, vocab, sentence, num_tokens=3, device='cpu'):
    """
    Given a sentence, predict the next `num_tokens` tokens.
    Uses the trained model to predict one token at a time.
    """
    # Create token-to-index mapping from the vocabulary.
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    model.eval()
    tokens = tokenizer.tokenize(vocab, sentence)
    context_size = model.embedding.weight.shape[1] if hasattr(model, 'embedding') else 4
    # We assume that the context_size used in training is the same as the one used here.
    # For prediction, if we have fewer than context_size tokens, we pad with "[PAD]".
    context = tokens[-context_size:]
    while len(context) < context_size:
        context = ["[PAD]"] + context

    predictions = []
    with torch.no_grad():
        for _ in range(num_tokens):
            # Convert context tokens to indices.
            context_indices = [token_to_idx.get(tok, token_to_idx.get("[UNK]")) for tok in context]
            context_tensor = torch.tensor([context_indices], dtype=torch.long, device=device)
            logits = model(context_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_token = vocab[pred_idx]
            predictions.append(pred_token)
            # Update context by sliding window (drop first token and append the predicted token)
            context = context[1:] + [pred_token]
    return predictions

# ----------------------------
# 5. Main Training Pipeline
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load corpus (modify the file path as needed)
    with open("corpus.txt", "r", encoding="utf-8") as f:
        # Assuming sentences are separated by ". "
        corpus = f.read().lower().split('\n')
    # Split into training and validation (80-20 split)
    split_idx = int(0.8 * len(corpus))
    train_corpus = corpus[:split_idx]
    val_corpus = corpus[split_idx:]

    # Initialize the tokenizer from Task 1
    tokenizer = WordPieceTokenizer()
    vocab_file = "vocabulary_88.txt"  # Path to vocabulary file
    # Load vocabulary to get size (same vocabulary used in Task 2)
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f.readlines()]
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    embedding_dim = 50  # Choose same dimension as in Task 2, or adjust accordingly
    context_size = 4    # For example, use 4 tokens as context

    # Create datasets and dataloaders
    train_dataset = NeuralLMDataset(train_corpus, tokenizer, vocab_file, context_size=context_size)
    val_dataset = NeuralLMDataset(val_corpus, tokenizer, vocab_file, context_size=context_size)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Load pretrained Word2VecModel embeddings if desired (here we simply initialize new embeddings)
    # For demonstration, we will initialize three different models.
    model1 = NeuralLM1(vocab_size, embedding_dim, context_size)
    model2 = NeuralLM2(vocab_size, embedding_dim, context_size)
    model3 = NeuralLM3(vocab_size, embedding_dim, context_size)

    # Train each model and collect histories
    print("Training NeuralLM1...")
    history1 = train_model(model1, train_loader, val_loader, epochs=10, lr=0.001, device=device)
    print("Training NeuralLM2...")
    history2 = train_model(model2, train_loader, val_loader, epochs=10, lr=0.001, device=device)
    print("Training NeuralLM3...")
    history3 = train_model(model3, train_loader, val_loader, epochs=10, lr=0.001, device=device)

    # Plot training and validation loss
    plot_history([history1, history2, history3], ["NeuralLM1", "NeuralLM2", "NeuralLM3"])

    # Compute final perplexity for each model (using final epoch validation loss)
    perplexity1 = compute_perplexity(history1['val_loss'][-1])
    perplexity2 = compute_perplexity(history2['val_loss'][-1])
    perplexity3 = compute_perplexity(history3['val_loss'][-1])
    print(f"Final Perplexities:\nNeuralLM1: {perplexity1:.4f}\nNeuralLM2: {perplexity2:.4f}\nNeuralLM3: {perplexity3:.4f}")

    # For demonstration, print final validation accuracies:
    print(f"Final Validation Accuracies:\nNeuralLM1: {history1['val_acc'][-1]:.4f}\nNeuralLM2: {history2['val_acc'][-1]:.4f}\nNeuralLM3: {history3['val_acc'][-1]:.4f}")

    # ----------------------------
    # 6. Next-Token Prediction on Test File
    # ----------------------------
    # Read test file (assumed one sentence per line)
    with open("sample_test.txt", "r", encoding="utf-8") as f:
        test_sentences = [line.strip() for line in f.readlines() if line.strip()]

    # Choose one of the trained models for prediction; here we use NeuralLM1 as an example.
    model_for_prediction = model1.to(device)
    predictions = {}
    for idx, sentence in enumerate(test_sentences):
        next_tokens = predict_next_tokens(model_for_prediction, tokenizer, vocab, sentence, num_tokens=3, device=device)
        predictions[f"sentence_{idx}"] = next_tokens

    # Save predictions to a JSON file
    with open("predicted_tokens.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=4)
    print("Next-token predictions saved to predicted_tokens.json.")

if __name__ == "__main__":
    main()