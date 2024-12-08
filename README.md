
# Genetic Algorithm for Decision Trees

This project implements a genetic algorithm to optimize decision trees. The genetic algorithm is used to select the best decision trees based on their accuracy and impurity.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contribution](#contribution)
- [License](#license)

## Introduction

The genetic algorithm is an optimization method inspired by natural selection. It is used here to optimize decision trees by selecting the best trees based on their accuracy and impurity.

## Installation

To use this project, you need to have Python installed on your machine. You can install the necessary dependencies using pip:

```bash
pip install numpy scikit-learn pandas graphviz
```

## Usage

Clone the repository:

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/your-repo.git
```

Navigate to the project directory:

Go to the project directory:

```bash
cd your-repo
```

Run the main script:

Run the main script to execute the genetic algorithm and evaluate the best decision tree:

```bash
python main.py
```

## Examples

Here is an example output you might get by running the main script:

```
Generation 1/20
Best fitness in generation: 0.8
Generation 2/20
Best fitness in generation: 0.82
...
Generation 20/20
Best fitness in generation: 0.85
Accuracy of the best tree on the test set: 0.85
Confusion Matrix:
[[50  5]
 [10 35]]
```

## Contribution

Contributions are welcome! If you would like to contribute to this project, follow these steps:

1. **Fork the repository**:
   - Click the "Fork" button at the top right of this page.
2. **Clone your fork**:
   - Clone your fork to your local machine:

     ```bash
     git clone https://github.com/your-username/your-repo.git
     ```
3. **Create a branch**:
   - Create a new branch for your modifications:

     ```bash
     git checkout -b name-of-your-branch
     ```
4. **Make changes**:
   - Make your changes and commit them:

     ```bash
     git add .
     git commit -m "Description of your changes"
     ```
5. **Push your changes**:
   - Push your changes to your fork:

     ```bash
     git push origin name-of-your-branch
     ```
6. **Create a Pull Request**:
   - Go to your fork on GitHub and click the "New Pull Request" button to create a Pull Request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
