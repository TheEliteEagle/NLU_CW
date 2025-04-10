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

This model has two parts: the MPNet base (a pretrained LLM) and the overlying classifier (an MLP).

- **Developed by:** Benjamin Hatton and Max Bolt
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Transformers
- **Finetuned from model [optional]:** bert-base-uncased

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


      - learning_rate: 1e-03
      - train_batch_size: 32
      - eval_batch_size: 1
      - seed: 42
      - num_epochs: 10

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->


      - overall training time: 5 hours
      - duration per training epoch: 30 minutes
      - model size: 300MB

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data & Metrics

#### Testing Data

<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

Around 6K text pairs provided within the task.

#### Metrics

<!-- These are the evaluation metrics being used. -->


      - Precision
      - Recall
      - F1-score
      - Accuracy

### Results

The model obtained an F1-score of 67% and an accuracy of 70%.

## Technical Specifications

### Hardware


      - RAM: at least 16 GB
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
