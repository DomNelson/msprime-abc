from keras.layers import Input, Dense
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
import msprime
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import attr
import sys, os
import pandas as pd
import argparse


def training_plot(history, outfile):
    """
    Plot training accuracy for each epoch
    """
    ## Set output file for plot
    basepath = os.path.split(os.path.expanduser(outfile))[0]
    plotfile = basepath + '_train_plot.png'

    ## Plot accuracy
    plt.plot(history.history['val_categorical_accuracy'], label='test')
    plt.plot(history.history['categorical_accuracy'], label='train')
    plt.ylim(0, 1)
    plt.legend()
    plt.savefig(plotfile)


@attr.s
class impute_network_sketch(object):
    """
    Container for network attributes
    """
    impute_indices = attr.ib()
    input_nodes = attr.ib()
    hidden_nodes = attr.ib()
    output_nodes = attr.ib()
    input_data = attr.ib()
    labels = attr.ib()


@attr.s
class Haplotypes(object):
    """
    Object for generating haplotypes with known IBD segments
    """
    length = attr.ib()
    rho = attr.ib()
    mu = attr.ib()
    N = attr.ib()
    Ne = attr.ib(default=10000)


    def __attrs_post_init__(self):
        self.ts = msprime.simulate(
                length=self.length,
                recombination_rate=self.rho,
                mutation_rate=self.mu,
                sample_size=self.N,
                Ne=self.Ne)


    def haplotypes(self):
        return self.ts.haplotypes()


    def positions(self):
        return (site.position for site in self.ts.sites())


    def breakpoints(self):
        return self.ts.breakpoints()


def main(args):

    data, labels = load_vcf(datafile, args.num_rows, args.num_cols,
                                                    args.impute_SNP)


    ## Balance data so that each category has equal representation
    data, labels = balance_data(data, labels)
    print(data.shape)

    ## Transform integers to categorical labels
    n_categories = len(set(labels))
    categorical_labels = to_categorical(labels)

    ## Get impute structure from one row of data
    N = sketch_network(data, categorical_labels)

    ## Layer which takes data as input
    data_input = Input(shape=(N.input_nodes,), name='input')

    ## Add a layer that biases weights towards nearby SNPs
    local_bias_layer = Dense(N.input_nodes, name='local_bias',
                        trainable=False, activation='linear')(data_input)

    ## Now add a hidden densely connected layer
    data_layer = Dense(N.input_nodes, name='hidden_1',
                                    activation='sigmoid')(local_bias_layer)

    ## Finally, the output layer gives a categorical prediction
    output_layer = Dense(categorical_labels.shape[1], name='output',
                                            activation='sigmoid')(data_layer)
    print(N.output_nodes)
    print(n_categories)

    ## Now create a model from the layers we have constructed
    model = Model(input=data_input, output=output_layer)

    ## Set weights for local bias layer
    lbl = model.get_layer(name='local_bias')
    bias = get_local_bias(args.impute_SNP, N.input_nodes)
    local_bias_weights = [np.identity(N.input_nodes), bias]
    lbl.set_weights(local_bias_weights)

    ## Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                                            metrics=['categorical_accuracy'])

    ## Save best model, by 'val_categorical_accuracy'
    callbacks = [ModelCheckpoint(args.outfile,
                            monitor='val_categorical_accuracy',
                            save_best_only=True)]
    ## Fit the model
    history = model.fit(N.input_data, categorical_labels, nb_epoch=args.epochs,
                            batch_size=1, validation_split=0.3, verbose=1,
                            callbacks=callbacks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required named arguments')

    requiredNamed.add_argument("-g", "--genotypes", metavar='|',
                        help="""File containing genotypes to train
                                imputation network""",
                        required=True)

    requiredNamed.add_argument("-r", "--num-rows", metavar='|',
                        help="Number of rows to read from file",
                        required=True, type=int)
    requiredNamed.add_argument("-c", "--num-cols", metavar='|',
                        help="Number of columns to read from file",
                        required=True, type=int)
    requiredNamed.add_argument("-o", "--outfile", metavar='|',
                        help="File to store trained model",
                        required=True)
    requiredNamed.add_argument("-s", "--impute-SNP", metavar='|',
                        help="SNP number to train for imputation",
                        required=True, type=int)
    requiredNamed.add_argument("-e", "--epochs", metavar='|',
                        help="Number of epochs to train network",
                        required=True, type=int)

    args = parser.parse_args()

    main(args)
