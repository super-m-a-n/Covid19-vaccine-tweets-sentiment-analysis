# covid19-vaccine-tweets-sentiment-analysis
# Project in Natural Language Processing
## About
In this project, we develop deep learning models for sentiment analysis classification of covid-19 vaccination tweets in to 3 categories: 
* pro-vax
* neutral
* anti-vax 

We develop and compare four different classifiers on the same datasets using real twitter data:

* Vaccine sentiment classifier using **softmax regression**
* Vaccine sentiment classifier using **feed forward neural networks**
* Vaccine sentiment classifier using **recurrent neural networks** 
* Vaccine sentiment classifier by fine-tuning the **pretrained BERT-base model**

## Softmax Regression Classifier
We experimented with :
* data cleaning / noise removal to prevent model from learning useless information / features than hinder the sentiment analysis from generalizing to unseen data
* different vectorizers (TF-IDF vectorizer, count vectorizer, Bag Of Words Vectorizing)
* max-df / min-df parameters to remove features with very low/high frequency
* regularization parameter to reduce overfitting
* Performance evaluated based on precision / recall / f1 score 


Model / experiments in : ``` /SoftmaxRegression_classifier/softmax_regression_model.ipynb ```

## Feed Forward Neural Network Classifier
We experimented with :
* data cleaning / noise removal to prevent model from learning useless information / features than hinder the sentiment analysis from generalizing to unseen data
* pre trained GloVe word embeddings
* different number of layers, different learning rates
* linear activation functions, non-linear activation functions (ReLU)
* dropout layers to reduce overfitting
* different optimizers (Stochastic gradient descent, Adam, RMSprop)
* Cross entropy loss function since we have multiclass unbalanced classification
* Performance evaluated based on precision / recall / f1 score / ROC curves


Model / experiments in : ``` /FFNN_classifier/feed_forward_nn_model.ipynb ```

## Recurrent Neural Network Classifier
We experimented with :
* Bidirectional stacked RNN's with LSTM/GRU cells
* Gradient Clipping to tackle the exploding gradients problem of RNN's
* Early stopping regularization to avoid overfitting (LSTM/GRU models learn much faster than previous models)
* Skip connections to train deeper RNN's and tackle the vanishing gradients problem
* Added attention to our best model to further improve performance
* Performance evaluated based on precision / recall / f1 score / ROC curves


Model / experiments in : ``` /RNN_classifier/recurrent_nn_model.ipynb ```

## Classifier by fine-tuning pre-trained BERT-base model
* BertForSequenceClassification model was used
* Experimented with the hyperparameters suggested in BERT's paper


Model / experiments in : ``` /BERT_classifier/bert_model.ipynb ```

