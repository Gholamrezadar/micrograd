from engine import Value
from nn import Mlp


n = Mlp(3, [4, 4, 1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

# Training
print("Iteration", "\tLoss")

for k in range(100):
    # Forward pass
    y_pred = [n(x) for x in xs]
    loss: Value = sum([(yout - ygt) ** 2 for ygt, yout in zip(ys, y_pred)])  # type: ignore

    # Backward pass
    n.zero_grad()
    loss.backward()

    # Update weights
    for p in n.parameters():
        p.data -= 0.03 * p.grad

    print(f"{str(k).center(len('Iteration'))}\t{loss.data :.5f}")

