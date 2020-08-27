from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
import pandas as pd


class Practice:
    def __init__(self):
        self.train = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
        self.test = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
        self.y_train = self.train.pop('survived')
        self.y_test = self.test.pop('survived')
        train_path = tf.keras.utils.get_file(
            "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
        test_path = tf.keras.utils.get_file(
            "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
        flower_CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
        self.flower_species = ['Setosa', 'Versicolor', 'Virginica']
        self.flower_train = pd.read_csv(train_path, names=flower_CSV_COLUMN_NAMES, header=0)
        self.flower_test = pd.read_csv(test_path, names=flower_CSV_COLUMN_NAMES, header=0)
        self.flower_y_train = self.flower_train.pop('Species')
        self.flower_y_test = self.flower_test.pop('Species')

    def basic_variables(self):
        string = tf.Variable('This is a string', tf.string)
        val = tf.Variable(3, tf.int16)
        floating = tf.Variable(5.1, tf.float64)
        print(string, val, floating)

    def titanic_lin_reg(self):

        def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
            def input_function():  # inner function, this will be returned
                ds = tf.data.Dataset.from_tensor_slices(
                    (dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
                if shuffle:
                    ds = ds.shuffle(1000)  # randomize order of data
                ds = ds.batch(batch_size).repeat(
                    num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
                return ds  # return a batch of the dataset

            return input_function  # return a function object for use

        cat_var = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
        num_var = ['age', 'fare']

        feature_cols = []
        train_test = [self.train, self.test]
        for cat in cat_var:
            vocab = self.train[cat].unique()
            feature_cols.append(tf.feature_column.categorical_column_with_vocabulary_list(cat, vocab))
        for num in num_var:
            feature_cols.append(tf.feature_column.numeric_column(num, dtype=tf.float32))

        train_input_fn = make_input_fn(self.train, self.y_train)
        test_input_fn = make_input_fn(self.test, self.y_test, num_epochs=1, shuffle=False)

        linear_est = tf.estimator.LinearClassifier(feature_columns=feature_cols)

        linear_est.train(train_input_fn)
        res = linear_est.evaluate(test_input_fn)

        print(res['accuracy'])

    def flowers(self):

        def input_fn(features, labels, training=True, batch_size=256):
            # Convert the inputs to a Dataset.
            dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

            # Shuffle and repeat if you are in training mode.
            if training:
                dataset = dataset.shuffle(1000).repeat()

            return dataset.batch(batch_size)

        f_cols = []
        for key in self.flower_train.keys():
            f_cols.append(tf.feature_column.numeric_column(key=key))

        classifier = tf.estimator.DNNClassifier(
            feature_columns=f_cols,
            hidden_units=[30, 10],
            n_classes=3)
        classifier.train(
            input_fn=lambda: input_fn(self.flower_train, self.flower_y_train),
            steps=5000
        )
        res = classifier.evaluate(
            input_fn=lambda: input_fn(self.flower_test, self.flower_y_test, training=False), steps=5000
        )
        print(res)


if __name__ == '__main__':
    ml = Practice()
    ml.flowers()
