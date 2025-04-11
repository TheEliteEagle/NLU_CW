
---

# Model Card for f08473bh-e27772mb-AV

<!-- Provide a quick summary of what the model is/does. -->



<!-- Provide a longer summary of what this model is. -->

This model works in two stages. 
First, there is a traditional feature extraction method, used to pick out stylometric, morphological, and syntactical features from the texts. 
Next there is a binary classifier that takes the data of the feature vectors and outputs a 1 or 0 based on its prediction. 
When training the model, a large amount of epochs was used, with memory so that the best performing model (on the evaluation set) was the one returned by the training algorithm.

The feature extraction comprises the following:
- Punctuation-based Features
  - Punctuation complexity: e.g. the number of instances of rarer punctuations (e.g. semicolons) as a percentage of total punctuation
  - Punctuation distribution
- POS-based Features
  - Ratio of determinants to nouns - relevant to some people often dropping articles in less formal setting, e.g. texts or personal emails
  - Proper capitalisation ratio - again gives insight into less formal writing
- Word-based Features
  - Range of word length - gives insight into the complexity of the language
  - Word length upper quartile - as mean and median both give relatively low values (as many common words are short) this gives more information as to the writer's common word length
  - Type-token ratio - gives insight into the writer's vocabulary diversity
- Sentence-based Features
  - Range of sentence length - gives insight into the common complexity of the sentence structures
  - Sentence length upper quartile - as this gives insight into the size of the writer's (generally) more complex senstences
- Readability-based Features
  - Flesch-Kincaid grade level - gives information on the quality and complexity of the writing
  - Typo ratio - gives insight into the writer's care and attention to detail
 
These were pre-calculated for each pair of texts to speed up training and evaluation. The feature vector for each text was 43 features long.
The binary classifier was a simple neural network-based model, an average sized MLP. 
To extract more features, the network was trained on 3 vectors of the same size: the vector for the first text, the vector for the second, and the absolute difference between the vectors.
The layers of the MLP are detailed below:
- Input layer: 43 * 3 (fv1, fv2, |fv1 - fv2|)
- Hidden layer 1: 128 neurons, ReLU activation
- Hidden layer 2: 64 neurons, ReLU activation
- Output layer: 1 neuron, no activation function

When training the network, the training algorithm was run for 200 epochs. The loss function was Binary Cross Entropy with Logits, and the optimiser was Adam (with a learning rate of 5e-5).
The 200 epoch model was liable to overfitting, therefore the current best performing model (in terms of accuracy on the evaluation set) was saved and returned by the training algorithm.

- **Developed by:** Benjamin Hatton and Max Bolt
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** Traditional ML
- **Finetuned from model [optional]:** n/a

### Model Resources

<!-- Provide links where applicable. -->

- **Repository:** n/a
- **Paper or documentation:** n/a

## Training Details



<!-- This is a short stub of information on the training data that was used, and documentation related to data pre-processing or additional filtering (if applicable). -->

27K pairs of text each with a label of 1 or 0 to denote whether it was written by the same person or not (respectively).

### Training Procedure


<!-- This is a summary of the values of hyperparameters used in training the model. -->


      - learning_rate: 5e-05
      - train_batch_size: 16
      - eval_batch_size: 1
      - seed: 42
      - num_epochs: 200

#### Speeds, Sizes, Times

<!-- This section provides information about how roughly how long it takes to train the model and the size of the resulting model. -->

      - feature vector calculation time (per 1000 text pairs): 98 seconds
      - overall training time: 3.75 minutes
      - duration per training epoch: 1.1 seconds
      - model size: 103.6 kB

## Evaluation



<!-- This should describe any evaluation data used (e.g., the development/validation set provided). -->

Around 6K text pairs with their labels provided specifically for the task.

#### Metrics



### Results

The model obtained an F1-score of 62.3% (with balanced recall and precision) and an accuracy of 62.03%. Further improvement could be made by exploring more features, and using a more complex model.
Features considered but not used (generally due to time constraints) include:
- Analysis of common mistakes, e.g. "your" vs "you're" and "there" vs "their" vs "they're"
- Common contractions, e.g. "can't", "won't", "isn't"
- Common abbreviations, e.g. "etc", "e.g.", "i.e."
- Connective usage (particularly if they are used in a non-standard way)
- Deeper analysis of typos, e.g. swapping letters, or missing letters (one shows fast typing, the other shows English knowledge)

## Technical Specifications

### Hardware

      - RAM: at least 2GB
      - Storage: at least 1GB,
      - GPU: n/a

### Software

      - SciPy and NumPy
      - pyspellchecker
      - textstat
      - Pytorch 1.11.0+cu113

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The main set of limitations within the model comes from the scope of features analysed. With expansion of the feature vector, the model could be improved.
Some bias could possibly be introduced from the Flesch-Kincaid grade level, as the vector is not normalised, and so there is a very large value being input into the network (when compared to the ratios etc.).

This model also should work reasonably well for texts of other languages, as the features (except typos) are not language specific.

There is little risk in using this model, as the data it uses does not keep its meaning after being processed into the feature vectors.

This model would work well as an extremely lightweight and fast way to get a rough idea about the shared authorship of two texts. Any more in depth analysis would have to be done by a deeper model, especially if the stakes are high. This would be best used as a way to get an idea if the authorship should be investigated more.

## Additional Information

