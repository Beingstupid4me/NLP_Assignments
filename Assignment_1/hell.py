import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from task1 import WordPieceTokenizer
from gensim.models import Word2Vec  # Importing Word2Vec from gensim

class NeuralLMDataset(Dataset):
    def __init__(self, file_path, context_size=3, tokenizer=None, word2vec_model=None):
        self.tokenizer = tokenizer if tokenizer else WordPieceTokenizer()  # Use custom or default tokenizer
        self.word_embeddings = word2vec_model.wv  # Access the word vectors from the Word2Vec model
        self.context_size = context_size
        self.data = []
        self.process_file(file_path)

    def process_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            corpus = f.readlines()
        
        for sentence in corpus:
            tokens = self.tokenizer.breakingOfTokens(sentence.strip())
            for i in range(len(tokens) - self.context_size):
                context_tokens = tokens[i:i+self.context_size]
                target_token = tokens[i+self.context_size]
                
                if target_token in self.word2vec_model:
                    input_embedding = torch.mean(torch.stack([
                        torch.tensor(self.word2vec_model[t], dtype=torch.float32) for t in context_tokens if t in self.word2vec_model
                    ]), dim=0)
                    target_embedding = torch.tensor(self.word2vec_model[target_token], dtype=torch.float32)
                    self.data.append((input_embedding, target_embedding))
                    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

class NeuralLM1(nn.Module):
    def __init__(self, embedding_dim=100):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)

class NeuralLM2(nn.Module):
    def __init__(self, embedding_dim=100):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

class NeuralLM3(nn.Module):
    def __init__(self, embedding_dim=100):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

def train(model, dataloader, epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")
def predict_next_words(model, sentence, top_k=3):
    tokenizer = word_tokenize  # Default tokenizer
    embeddings = word2vec_model.wv  # Use Word2Vec model's word vectors
    
    tokens = tokenizer(sentence.strip())
    input_embedding = torch.mean(torch.stack([
        torch.tensor(embeddings[t], dtype=torch.float32) for t in tokens if t in embeddings
    ]), dim=0)

    output_embedding = model(input_embedding.unsqueeze(0))
    
    # Convert numpy.ndarray to torch.Tensor for embeddings
    embedding_tensor = torch.tensor(embeddings.vectors, dtype=torch.float32)
    similarities = torch.matmul(output_embedding, embedding_tensor.T)
    
    top_indices = torch.argsort(similarities, descending=True)[0][:top_k]
    
    predicted_words = [list(embeddings.index_to_key)[i] for i in top_indices]
    return predicted_words

# Load the corpus and train the Word2Vec model
with open("corpus.txt", 'r', encoding='utf-8') as f:
    corpus = f.readlines()

# Tokenize corpus and train Word2Vec model using gensim
tokenized_corpus = [word_tokenize(sentence.strip()) for sentence in corpus]
word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)

# Load dataset (make sure to use the trained Word2Vec model)
dataset = NeuralLMDataset("corpus.txt", tokenizer=word_tokenize, word2vec_model=word2vec_model)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Train models
for ModelClass in [NeuralLM1, NeuralLM2, NeuralLM3]:
    print(f"Training {ModelClass.__name__}")
    model = ModelClass()
    train(model, dataloader)

# Test next-word prediction
with open("sample_test.txt", 'r', encoding='utf-8') as f:
    test_sentences = f.readlines()

for sentence in test_sentences:
    print(f"Input: {sentence.strip()} \nPredicted: {predict_next_words(model, sentence)}\n")
