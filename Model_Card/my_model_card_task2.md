---
{}
---
language: en
license: cc-by-4.0
tags:
- text-classification
repo: https://github.com/TheEliteEagle/NLU_CW

---

# Model Card for f08473b-e27772mb-AV

<!-- Provide a quick summary of what the model is/does. -->

This is a classification model that was trained to
      detect whether two pieces of text were written by the same author.


## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model has two parts: the MPNet base (a pretrained LLM) and an ensemble of binary classifiers. Each binary classifier has 3 layers, of 1024, 512 and 1, each with a Relu, Layernorm and Dropout layer between them.

- **Developed by:** Benjamin Hatton and Max Bolt
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Deep-learning ensemble
- **Finetuned from model [optional]:** all-mpnet-base-v2

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** https://huggingface.co/sentence-transformers/all-mpnet-base-v2
- **Paper or documentation:** https://arxiv.org/pdf/2004.09297

## Training Details

### Training Data

<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

27K pairs of text each with a label of 1 or 0 to denote whether it was written by the same person or not (respectively).

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Training Hyperparameters

<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - children: 5
      - learning_rate: 1e-03
      - train_batch_size: 64
      - eval_batch_size: 1
      - seed: 42
      - num_epochs: 40

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 20 minutes
      - duration per training epoch: 0.5 minutes
      - model size: 72mb

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

Around 6K text pairs provided within the task.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision: 0.68
      - Recall: 0.67
      - F1-score: 0.68
      - Accuracy: 0.68

### Results

The model obtained an F1-score of 68% and an accuracy of 67%.

## Technical Specifications

### Hardware


      - RAM: at least 8 GB
      - Storage: at least 2GB,
      - GPU: V100

### Software


      - Transformers 4.18.0
      - Pytorch 1.11.0+cu113

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Any inputs (concatenation of two sequences) longer than
      512 subwords will be truncated by the model.

## Additional Information

<!-- Any other information that would be useful for other people to know. -->

The hyperparameters were determined by experimentation
      with different values.
