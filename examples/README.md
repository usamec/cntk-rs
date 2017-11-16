## MNIST examples (feedforward net and conv net)

### Preparation

Download and extract into `data` directory: http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz,
http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz, http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz and
http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz.

### Running

Fully connected network:
`cargo run --example mnist_mlp`

Convolutional network: `cargo run --example mnist_convnet`

## Text example (word embedings and seq2seq)

### Preparation
Download into `data` directory: http://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt

### Running

Calculating word embeddings.
`cargo run --example sparse_ops_and_word_embeddings`

Simple sequence autoencoder with GRU units.
`cargo run --example seq2seq`

## Other examples

### Quasi recurrent bidirectional neural network

This network combines convolutional network with recurrent.

`cargo run --example qrnn`