# lstm-aya
LSTM model implementation based off of Denny Britz's GRU model.
Go to https://github.com/dennybritz/rnn-tutorial-gru-lstm for his original code.

This needs to be run on GPU. I used Amazon Web Services g2.2xlarge instance. The training time takes approximately 3 hours.

The code is writen in Python and uses the Theano library. 
Make sure you have Python 3 and Theano installed. This is a nice blog post that helps with setting up Theano on AWS.
http://markus.com/install-theano-on-aws/

To run:
python train.py

There is a dataset of 700 Japanese lyrics in the data file called input.txt. Note that this is not a large enough dataset.
The train.py will automatically generate an output in the end as well as save the model parameters to a .npz file.

I modified the forward propagation, preprocessing (load_data method) to tokenize Japanese words using Masata Hagiwara's tinysegmenter.

The code does generate a sample output but the results are pretty bad. Still in the process of fixing the generate method which is giving me errors when I change the parsing from taking in 1 lyrics as a input instance to 1 line as an instance (in order to increase the size of the dataset).


