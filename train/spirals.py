import numpy as np
import matplotlib.pyplot as plt

seed = 123
np.random.seed(seed)

def generate_spiral_dataset(n_samples, n_classes, noise=0.12):
    angle = 2 * np.pi / n_classes
    data = []
    labels = []

    for class_idx in range(n_classes):
        radius = np.linspace(0.1, 1, num=n_samples//n_classes)
        theta = np.linspace(class_idx * angle, (class_idx + 1) * angle,
                            num=n_samples//n_classes) + np.random.randn(n_samples//n_classes) * noise
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        class_data = np.column_stack((x, y))
        data.append(class_data)
        labels.extend([class_idx] * (n_samples//n_classes))

    data = np.concatenate(data, axis=0)
    labels = np.array(labels)

    return data, labels

# Generate spiral dataset with 500 samples and 3 classes
if __name__ == "main":
    n_samples = 500
    n_classes = 3
    data, labels = generate_spiral_dataset(n_samples, n_classes)
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.title('Spiral Dataset')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig('Spirals visualized.png')
