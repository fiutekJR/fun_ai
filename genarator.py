import torch
from torch import nn
from collections import Counter
import pickle

with open('dataset/jokes.txt', 'r', encoding='utf-8') as f:
    jokes = f.read().split('\n')

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

class JokesModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

# Создание экземпляра модели (такого же, как тот, который вы обучали)
model = JokesModel(len(vocab), 64, 128, 2)

# Загрузка весов модели
model.load_state_dict(torch.load('jokes_model.pth'))

# Перевод модели в режим оценки (это отключает некоторые механизмы, такие как dropout, которые используются только во время обучения)
model.eval()


def generate_joke(model, vocab, seed_text, max_length=50):
    model.eval()
    joke = [vocab[word] for word in seed_text.split()]
    for _ in range(max_length):
        input = torch.tensor([joke]).long()
        output = model(input)
        next_word = output.argmax(dim=2)[:, -1].item()
        joke.append(next_word)
    inv_vocab = {v: k for k, v in vocab.items()}
    return ' '.join(inv_vocab[word] for word in joke)



print(generate_joke(model, vocab, "Почему курица"))
