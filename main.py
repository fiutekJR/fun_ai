import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torch.nn.functional import one_hot
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

epochs = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Загрузка CSV-файла
df = pd.read_csv('dataset/short_jokes75k.csv')

# Извлечение колонки с шутками
jokes_series = df['text']
ratings = df['rate']

# Преобразование Series в список
jokes = [str(joke) for joke in jokes_series.tolist()]

# Построение словаря
counter = Counter(' '.join(jokes).split())
vocab = {word: i for i, word in enumerate(counter.keys())}
vocab['<PAD>'] = len(vocab)

with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)

# Подготовка данных
class JokesDataset(Dataset):
    def __init__(self, jokes, vocab):
        self.jokes = [torch.tensor([vocab[word] for word in joke.split()]) for joke in jokes if joke.split()]
    
    def __len__(self):
        return len(self.jokes)
    
    def __getitem__(self, idx):
        return self.jokes[idx]

def collate_fn(batch):
    return pad_sequence(batch, padding_value=vocab['<PAD>'])

dataset = JokesDataset(jokes, vocab)
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# Создание модели
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

model = JokesModel(len(vocab), 64, 128, 2)
model = model.to(device)

# Компиляция и обучение модели
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

for epoch in range(epochs):
    total_loss, total_correct, total_count = 0, 0, 0
    for i, jokes in enumerate(dataloader):
        jokes = jokes.to(device)
        optimizer.zero_grad()
        input = jokes[:, :-1]
        target = jokes[:, 1:]
        output = model(input)
        loss = criterion(output.transpose(1, 2), target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (output.argmax(dim=2) == target).sum().item()
        total_count += target.numel()
        print(f'Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}')
    history['loss'].append(total_loss / len(dataloader))
    history['accuracy'].append(total_correct / total_count)
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}, Accuracy: {total_correct / total_count:.4f}')



# Построение графиков
epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['accuracy'], label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['loss'], label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.show()

# Сохранение модели
torch.save(model.state_dict(), 'jokes_model.pth')
print('Model saved')

# Генерация шутки
def generate_joke(model, vocab, seed_text, max_length=50):
    model.eval()
    joke = [vocab[word] for word in seed_text.split()]
    for _ in range(max_length):
        input = torch.tensor([joke]).long().to(device)
        output = model(input)
        next_word = output.argmax(dim=2)[:, -1].item()
        if next_word == vocab['<PAD>']:
            break
        joke.append(next_word)
    inv_vocab = {v: k for k, v in vocab.items()}
    return ' '.join(inv_vocab[word] for word in joke)


print(generate_joke(model, vocab, "Почему курица"))
