import torch  # PyTorch main package
from torch import nn  # neural network layers and loss functions
from torch.utils.data import DataLoader  # data loader to create mini-batches
from torch.nn.utils.rnn import pad_sequence  # utility to pad variable-length sequences
from datasets import load_dataset  # Hugging Face datasets loader for IMDB
import re  # regular expressions for tokenization
from collections import Counter  # counter to build vocabulary by frequency


# ----- Load the data
dataset = load_dataset("imdb")       # returns {'train': Dataset, 'test': Dataset}
train_data = dataset['train']
test_data = dataset['test']


# ----- Create a dictionary with the most of the words used

# finding words in text
def tokenizer(text):
    return re.findall(r'\b\w+\b', text.lower())

# identify frequent words and create a dictionary with them
def build_vocab(data, min_freq=5):
    counter = Counter()
    for example in data:
        counter.update(tokenizer(example['text']))
    vocab = {"<unk>": 0, "<pad>": 1}
    i = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = i
            i += 1
    return vocab

vocab = build_vocab(train_data)  # create vocabulary from the training data
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")


# ----- Initialising the data pipeline
MAX_LEN = 200  # maximum sequence length (longer sequences will be truncated)

# converting raw text -> tensor of token indices
def text_pipeline(x):
    tokens = [vocab.get(t, vocab["<unk>"]) for t in tokenizer(x)][:MAX_LEN]
    return torch.tensor(tokens, dtype=torch.long)

# assure label that is boolean
def label_pipeline(x):
    return 1 if x == 1 else 0

# compress the batch
def collate_batch(batch):
    label_list = []
    text_list = []
    for example in batch:
        label_list.append(label_pipeline(example['label'])) # convert label
        text_list.append(text_pipeline(example['text']))  # convert text
    text_list = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"]) # padding the tensors
    label_list = torch.tensor(label_list, dtype=torch.long)  # labels as long integers
    return text_list, label_list


# ----- DataLoader
batch_size = 32

# training DataLoader will shuffle each epoch and use collate_batch to produce tensors
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)


# ----- Defining the model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=1) # transform IDs in trainable vectors
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True) # processes the sequence, remembering important info
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        embedded = self.embedding(text)
        _, (hidden, _) = self.lstm(embedded)  # output, (hidden, memory_cell) = lstm(input)
        out = self.fc(hidden.squeeze(0)) # hidden.shape = (1, batch, hidden_dim) => (batch, hidden_dim) after squeeze
        return self.sigmoid(out)

embed_dim = 128
hidden_dim = 128
output_dim = 1

model = SentimentLSTM(vocab_size, embed_dim, hidden_dim, output_dim)


# ----- Training setup
criterion = nn.BCELoss()  # binary cross-entropy loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# ----- Training
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for text, labels in train_dataloader:
        text = text.to(device)
        labels = labels.to(device).float()
        optimizer.zero_grad()
        output = model(text).squeeze()  # squeeze last dim so the shape => (batch,)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # print average loss per batch for this epoch
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader):.4f}")


# ----- Evaluating
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for text, labels in test_dataloader:
        text = text.to(device)
        labels = labels.to(device).float()
        output = model(text).squeeze()  # squeeze last dim so the shape => (batch,)
        predicted = (output >= 0.5).long()  # convert probabilities to predicted labels 0/1

        correct += (predicted == labels.long()).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {correct/total:.4f}")
