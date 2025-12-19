import random

def generate_sequence(length=50):
    """
    Long-range dependency:
    x_t = x_{t-3} XOR x_{t-10}

    Previous token alone gives no information.
    """
    seq = [random.randint(0, 1) for _ in range(10)]

    for t in range(10, length):
        next_token = seq[t-3] ^ seq[t-10]  # XOR dependency
        seq.append(next_token)

    return seq


def generate_dataset(n_sequences=1000, length=50):
    return [generate_sequence(length) for _ in range(n_sequences)]
