import os
import numpy as np
import random
import pickle as pkl

def bits_to_vector(bits):
    """
    Given bit input (string), convert to 8x1 vector
    of floats
    """
    vec = np.zeros((len(bits),1))
    for bit_index, bit in enumerate(bits):
        vec[bit_index] = bit
    return vec

def read_test_data():
    """
    Read in 8-bit autoencoder data from
    file auto_encoder_input.txt and return
    as list of pairs of strings
    """
    cwd = os.getcwd()
    sequence = []
    filename = './auto_encoder_input.txt'
    test_data = []
    with open(filename, "r") as f:
        for line in f:
            test_data.append((line.replace('\n', ''), line.replace('\n', '')))

    return test_data

def training_data_to_vecs(test_data):
    """
    Takes 8-bit training data (in file auto_encoder_input.txt) and 
    turns into list of pairs of 8-bit vectors
    """
    test_data_vecs = []
    for x,y in test_data:
        x_vec = bits_to_vector(x)
        y_vec = bits_to_vector(y)
        test_data_vecs.append((x_vec,y_vec))
    return test_data_vecs

def read_DNA_training_data():
    """
    Read in positive binding site sequences from file
    rap1-lieb-positives.txt and return as list of
    pairs of strings (input and output strings are the
    same)
    """
    cwd = os.getcwd()
    sequence = []
    filename = './rap1-lieb-positives.txt'
    data = []

    with open(filename, "r") as f:
        for line in f:
            data.append((line.replace('\n', ''), line.replace('\n', '')))
    return data


def DNA_to_vector(DNA_string):
    """
    Encode DNA string as vector with scheme
    A: 1000
    G: 0100
    T: 0010
    C: 0001
    """

    vec = []
    for c in DNA_string:
        if c == 'A':
            vec.append([1.0])
            vec.append([0.0])
            vec.append([0.0])
            vec.append([0.0])
        if c == 'G':
            vec.append([0.0])
            vec.append([1.0])
            vec.append([0.0])
            vec.append([0.0])
        if c == 'T':
            vec.append([0.0])
            vec.append([0.0])
            vec.append([1.0])
            vec.append([0.0])
        if c == 'C':
            vec.append([0.0])
            vec.append([0.0])
            vec.append([0.0])
            vec.append([1.0])
    return(np.array(vec))


def DNA_training_data_to_vecs(DNA_training_data):
    """
    Converts sequence training data (in sequence String form)
    to vector form, returns list of pairs of bits
    """
    DNA_data_vecs = []
    for x,y in DNA_training_data:
        x_vec = DNA_to_vector(x)
        y_vec = DNA_to_vector(y)
        DNA_data_vecs.append((x_vec,y_vec))
    return DNA_data_vecs

def read_test_DNA():
    """
    Read in sequences from file rap1-lieb-test.txt 
    and return list of strings
    """
    cwd = os.getcwd()
    filename = './rap1-lieb-test.txt'
    test_data = []
    with open(filename, "r") as f:
        for line in f:
            test_data.append((line.replace('\n', '')))
    return test_data

def read_negative_DNA(positive_DNA_sequence_pairs, pickle = True):
    """
    Read in the negative binding site sequences from file yeast-upstream-1k-negative.fa
    """
    if pickle:
        negative_sequences = pkl.load(open('negative_sequences.pkl', 'rb'))
        return(negative_sequences)

    else:
        cwd = os.getcwd()
        filename = 'yeast-upstream-1k-negative.fa'
        sequences = []
        positive_sequences = [x for (x,y) in positive_DNA_sequence_pairs]
        with open(filename, "r") as f:
            current_sequence = ''
            for line in f:
                if line[0] == '>':
                    sequences.append(current_sequence)
                    current_sequence = ''
                else:
                    current_sequence = current_sequence + line.replace('\n', '')

        negative_sequences = []
        for sequence in sequences:
            discard = False
            sub_sequences = []
            for i in range(len(sequence) - 18):
                sub_seq = sequence[i : i + 17]
                sub_sequences.append(sub_seq)
                if sub_seq in positive_sequences:
                    discard = True
            if not discard:
                for sub_seq in sub_sequences:
                    negative_sequences.append((sub_seq, sub_seq))
                # negative_sequences.append(sequence)

        pkl.dump(negative_sequences, open('negative_sequences.pkl', 'wb'))
        return(negative_sequences)


def create_training_data(pickle = True):
    """
    Creates a master training set by combining known positive and 
    negative sequences. Includes all positive sequences from file 
    rap1-lieb-positives.txt. Chooses negative sequences to include
    by taking all 1000bp negative sequences and looking at all
    substrings of length 17 in each once. If the 1000bp sequence 
    contains a substring which is a positive sequence, discard
    that 1000bp sequence. If not, add all 17-bp substrings
    to a master set of negative sequences. Once this master
    set is created, choose a random subset to include in the 
    final master training set such that the final ratio of 
    negative to positive sequences in the training set is
    10:1. 
    """
    
    if pickle:
        training_set = pkl.load(open('training_data.pkl', 'rb'))
        return training_set

    else:
        positive_sequences = read_DNA_training_data()
        negative_sequences = read_negative_DNA(positive_sequences)
        
        training_set = []
        for seq , _ in positive_sequences:
            input_vec = DNA_to_vector(seq)
            output_vec = np.array([1.0])
            training_set.append((input_vec, output_vec))

        num_positive_sequences = len(positive_sequences)

        negative_sample_number = 10 * num_positive_sequences
        sampling_indices = []
        for i in range(len(negative_sequences)):
            sampling_indices.append(i)

        random.shuffle(sampling_indices)
        sampling_indices = sampling_indices[0 : negative_sample_number]

        for index in sampling_indices:
            seq, _ = negative_sequences[index]
            input_vec = DNA_to_vector(seq)
            output_vec = np.array([0.0])
            training_set.append((input_vec, output_vec))

        pkl.dump(training_set, open('training_data.pkl', 'wb'))
        return training_set
