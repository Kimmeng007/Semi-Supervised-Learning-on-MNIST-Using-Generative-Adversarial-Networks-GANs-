# Semi-Supervised Learning on MNIST using GANs

This project explores **semi-supervised learning** using **Generative Adversarial Networks (GANs)** under a highly constrained labeling budget. We implement and analyze a semi-supervised GAN inspired by *Salimans et al. (2016)* and compare its performance against a purely supervised baseline on the **MNIST handwritten digit dataset**. The key idea is to leverage **unlabeled data** to improve classification performance when labeled data is scarce.

## Project Overview

- **Task**: Digit classification on MNIST
- **Setting**: Semi-supervised learning
- **Labeled data**: As few as 10–100 samples
- **Unlabeled data**: Remaining MNIST training set
- **Models**:
  - Semi-supervised GAN (K + 1 classifier)
  - Supervised baseline (K-class classifier)

## Methodology

### Semi-Supervised GAN

We follow the framework proposed by **Salimans et al. (2016)**:

- The **discriminator** is modified to output **K + 1 classes**:
  - K real digit classes (0–9)
  - 1 additional *fake* class for generated samples
- The **generator** produces synthetic images from random noise
- The discriminator is trained using:
  - **Supervised loss** on labeled data
  - **Unsupervised loss** on unlabeled real data and generated samples

### Feature Matching

Instead of directly fooling the discriminator, the generator is trained using **feature matching**, encouraging generated samples to match the statistics of real data in an intermediate discriminator layer. This stabilizes training and improves representation learning.

## Model Architecture

### Discriminator (Classifier)

| Layer | Output Shape |
|------|-------------|
| Input | 1 × 28 × 28 |
| Conv (32, stride 2) | 32 × 14 × 14 |
| Conv (64, stride 2) | 64 × 7 × 7 |
| Fully Connected | 128 |
| Output | K + 1 |

### Generator
- Fully connected network
- Maps random noise vectors to 28×28 grayscale images

## Training Details
- **Optimizer**: Adam
- **Learning Rate**:
  - GAN: `2e-4`
  - Baseline: `1e-3`
- **β parameters**: (0.5, 0.999)
- **Epochs**: Fixed number (no early stopping)
- **Evaluation metric**: Test accuracy on MNIST

## Experiments
- Label budgets: **10 to 100 labeled samples**
- Multiple runs per setting with different random label subsets
- Reported results:
  - Mean test accuracy
  - Standard deviation

Both models share the **same architecture and preprocessing**, ensuring a fair comparison.

## Results
Key observations:
- In the **low-label regime (10–30 labels)**:
  - Semi-supervised GAN performs comparably to the supervised baseline
- In the **mid-label regime (40–60 labels)**:
  - Semi-supervised GAN consistently outperforms the baseline
- In the **high-label regime (80–100 labels)**:
  - Performance converges for both models

## Conclusion

This project confirms that **semi-supervised GANs** can effectively leverage unlabeled data to improve classification performance when labels are scarce. While gains on MNIST are moderate, the observed trends align with theory and prior work, highlighting the potential of GAN-based semi-supervised learning.

## Future Work

- Apply the method to more complex datasets
- Explore alternative semi-supervised objectives
- Improve discriminator architectures
- Investigate scalability and stability improvements

## References

- T. Salimans et al., *Improved Techniques for Training GANs*, NeurIPS 2016
