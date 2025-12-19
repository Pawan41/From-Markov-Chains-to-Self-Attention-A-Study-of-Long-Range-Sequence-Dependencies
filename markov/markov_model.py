import numpy as np

class MarkovModel:
    def __init__(self):
        self.transition = np.zeros((2, 2))

    def train(self, sequences):
        for seq in sequences:
            for i in range(len(seq) - 1):
                self.transition[seq[i], seq[i+1]] += 1

        self.transition /= self.transition.sum(axis=1, keepdims=True)

    def predict_next(self, prev_token):
        return np.argmax(self.transition[prev_token])

    def evaluate(self, sequences):
        correct, total = 0, 0
        for seq in sequences:
            for i in range(len(seq) - 1):
                pred = self.predict_next(seq[i])
                if pred == seq[i+1]:
                    correct += 1
                total += 1
        return correct / total
