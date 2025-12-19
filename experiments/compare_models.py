# -----------------------------
# Imports
# -----------------------------
import torch
import torch.nn as nn

from data.synthetic_sequences import generate_dataset
from markov.markov_model import MarkovModel
from transformer.mini_transformer import MiniTransformer


# -----------------------------
# Helper function (ADD HERE)
# -----------------------------
def batchify(data, batch_size=32):
    """
    Converts list of sequences into mini-batches
    """
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        x = torch.tensor([seq[:-1] for seq in batch], dtype=torch.long)
        y = torch.tensor([seq[1:] for seq in batch], dtype=torch.long)
        yield x, y


# -----------------------------
# Data
# -----------------------------
train_data = generate_dataset(3000)
test_data = generate_dataset(500)


# -----------------------------
# MARKOV MODEL
# -----------------------------
markov = MarkovModel()
markov.train(train_data)
markov_acc = markov.evaluate(test_data)


# -----------------------------
# TRANSFORMER
# -----------------------------
model = MiniTransformer(embed_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()


# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(200):
    total_loss = 0

    for x, y in batchify(train_data, batch_size=32):
        logits, _ = model(x)

        # Predict ONLY last token
        loss = loss_fn(logits[:, -1, :], y[:, -1])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")


# -----------------------------
# Evaluation
# -----------------------------
correct, total = 0, 0
for seq in test_data:
    x = torch.tensor(seq[:-1], dtype=torch.long).unsqueeze(0)
    y = seq[-1]

    preds, _ = model(x)
    pred = preds[:, -1, :].argmax(-1).item()

    correct += int(pred == y)
    total += 1

transformer_acc = correct / total


# -----------------------------
# Results
# -----------------------------
print(f"Markov Accuracy:      {markov_acc:.3f}")
print(f"Transformer Accuracy: {transformer_acc:.3f}")

