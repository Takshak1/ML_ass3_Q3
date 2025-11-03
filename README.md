1️⃣ Next-Word Prediction using MLP — (5 marks)

Extend the next-character prediction notebook (discussed in class) to a next-word prediction problem using an MLP-based text generator.
Train the model, visualize learned word embeddings, and deploy a Streamlit app for interactive text generation.
You must use two datasets — one from each category below:

Category I (Natural Language)	Category II (Structured/Domain Text)
Paul Graham Essays	Maths textbook
Wikipedia (English)	Python/C++ code (e.g., Linux Kernel)
Shakespeare	IITGN advisory generation
War and Peace (Leo Tolstoy)	IITGN website text
Sherlock Holmes (Conan Doyle)	sklearn docs / notes / ASCII art / music generation
1.1 Preprocessing and Vocabulary Construction — (0.5 mark)

Remove special characters except "." for sentence splitting.

import re
line = re.sub('[^a-zA-Z0-9 \.]', '', line)


Convert text to lowercase.

Build a vocabulary using unique words.

Report:

Vocabulary size

10 most frequent and 10 least frequent words

Create (X, y) pairs similar to next-character prediction (e.g., "..... -> to" at paragraph change).

1.2 Model Design and Training — (1 mark)

Architecture:

Embedding dimension: 32 or 64

Hidden layers: 1–2 (1024 neurons each)

Activation: ReLU or Tanh

Output: Softmax over vocabulary

Training Setup:

Run for 500–1000 epochs (use Colab or Kaggle)

Report:

Training vs validation loss plot

Final validation loss/accuracy

Example text generations + commentary on learning behavior

1.3 Embedding Visualization and Interpretation — (1 mark)

Use t-SNE for >2D embeddings or scatter plots for 2D.

Select meaningful word sets (synonyms, antonyms, verbs/adverbs, etc.)

Discuss clustering patterns and semantic relationships observed in the embeddings.

1.4 Streamlit Application — (1.5 marks)

Develop a Streamlit app that:

Accepts user input text

Predicts next k words or lines

Has controls for:

Context length

Embedding dimension

Activation function

Random seed

Temperature control for randomness

Handles unknown user words gracefully (no retraining required)

Report:

Provide a link to your deployed Streamlit app.

Offer 2–3 model variants as selectable options.

1.5 Comparative Analysis — (1 mark)

Compare your two trained models (Natural vs Structured dataset):

Aspect	Category I	Category II
Dataset size		
Vocabulary		
Context predictability		
Model performance		
Embedding visualization		

Discussion:
Summarize insights on how natural vs structured language differ in learnability.

2️⃣ Moons Dataset & Regularization — (3 marks)

Create and analyze models on a synthetic 2D dataset to study regularization and robustness.

2.1 Dataset

Generate your own Make-Moons dataset (not from sklearn).

Noise levels: 0.2 (train), 0.1 and 0.3 (test robustness).

500 points each for train and test sets.

Standardize x after split (using train statistics).

Validation: 20% of training set.

Random seed: 1337.

2.2 Models to Train

MLP (with hidden layer + early stopping, patience=50)

MLP with L1 regularization

Grid λ ∈ {1e−6, 3e−6, 1e−5, 3e−5, 1e−4, 3e−4}

Report layerwise sparsity and validation AUROC vs λ

MLP with L2 regularization (tune λ using validation set)

Logistic Regression with polynomial features (x₁x₂, x₁², etc.)

2.3 Evaluation & Analysis

Test accuracy on noise levels 0.2, 0.1, and 0.3

Create table:

Model	Params	Test Acc (0.1)	Test Acc (0.2)	Test Acc (0.3)
...	...	...	...	...

Plot decision boundaries for all 4 models (noise=0.2)

Discuss:

Effect of L1 on sparsity and boundary jaggedness

Effect of L2 on smoothness and margin

Bonus: Add class imbalance (70:30) and report effect on accuracy & AUROC.

3️⃣ MNIST and CNN Experiments — (3 marks)

This section explores deep learning for images using MLPs and CNNs on the MNIST dataset, and compares with baselines.

3.1 Using MLP — (1.5 marks)

Train an MLP with:

30 neurons (1st layer)

20 neurons (2nd layer)

10 output neurons (classes)

Compare against:

Random Forest

Logistic Regression

Metrics: Accuracy, F1-score, Confusion Matrix

Report:

Discuss observations & misclassifications

Visualize t-SNE of 20-neuron layer for trained vs untrained models

Test trained MLP on Fashion-MNIST and compare t-SNE plots for MNIST vs Fashion-MNIST embeddings

3.2 Using CNN — (1.5 marks)

Build and compare:

Simple CNN

Conv layer (32 filters, 3×3)

MaxPool layer

FC layer (128 neurons)

Output layer (10 neurons)

Activation: ReLU

Two pretrained CNNs (e.g., MobileNet, ResNet, EfficientNet)
