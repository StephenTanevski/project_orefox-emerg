"""Holds a class for analysing data."""

import math
from sys import getsizeof
import time
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers
import xgboost as xgb


class Analyser:
    """Holds methods for analysing data from a datacleaner.
    
    Attributes:
        datacleaner: A DataCleaner that holds the data to perform analysis on.
        logger: The logger from datacleaner to log events.
        kmeans: An instance of sklearn.cluster.KMeans.
        rf: An instance of sklearn.ensemble.RandomForestRegressor.
        pca: An instance of sklearn.decomposition.PCA.
        fa: An instance of sklearn.decomposition.FactorAnalysis.
        hca: Holds the return of linkage from scipy HCA.
        stats: Holds preliminary stats of the data.
    """

    # Initialise these variables now
    kmeans = None
    rf = None
    pca = None
    fa = None
    hca = None
    stats = None
    correlations = None
    adaboost = None
    xgboost = None

    def __init__(self, datacleaner):
        """Create an Analyser.
        
        Args:
            datacleaner: The DataCleaner to perform analysis on.
        """
        self.datacleaner = datacleaner

        # Get the logger as a member of this class to make logging calls
        # less complicated
        self.logger = datacleaner.logger

        self.logger.logger.info('Created Analyser')


    def get_dataset_stats(self) -> float:
        """Gets overall stats of data.

        Dictionary is saved as the following set of (key, value) pairs:
            num_features: The number of columns in the dataset.
            num_entries: The number of rows, not including header.
            num_empty: The number of empty data entries.
            empty_cells_per_row: A dictionary of rows: num_empty pairs where
              the value is specifically the number of empty rows in that row.
            empty_cells_per_column: Same as above but column wise.
            data_size: The return of getsizeof(data).

        Returns:
            Time taken to complete this method.
        """
        # Get the start time
        start_time = time.time()

        stats = {}

        # Store number of columns and rows in the data
        stats['num_features'] = self.datacleaner.data.shape[1]
        stats['num_entries'] = self.datacleaner.data.shape[0]

        # Count empty cells 
        empty_coords = np.where(pd.isnull(self.datacleaner.data))
        stats['num_empty'] = len(empty_coords[0])

        # Initialise all rows to 0
        rows = {}
        for row in range(self.datacleaner.data.shape[0]):
            rows[row] = 0

        # Now count the number of empty cells per row
        for row in empty_coords[0]:
            rows[row] += 1

        stats['empty_cells_per_row'] = rows

        # Now do the same for columns
        columns = {}
        for column in range(self.datacleaner.data.shape[1]):
            columns[column] = 0

        for column in empty_coords[1]:
            columns[column] += 1

        stats['empty_cells_per_column'] = columns

        # Size of data
        stats['data_size'] = getsizeof(self.datacleaner.data)

        # Log success
        self.logger.logger.info("get_dataset_stats success")

        self.stats = stats
        return time.time() - start_time


    def summarise_float_column(self, column):
        """
        Returns the pandas description of a given column.

        Args:
            column: The column to get the description of.
        """
        return self.datacleaner.data[column].describe()

    
    def get_columns_proportional_missing(self, proportion: float=None, 
            num_missing: int=None, drop_columns: bool=False):
        """Gets columns that have a certain proportion/number of empty cells.
        
        Args:
            proportion: A value in [0, 1] representing the proportion of empty
              cells compared to the total number of rows, that if exceeded,
              should be returned (and potentially dropped).
            num_missing: The number of missing cells that must be in a column
              to be returned, and potentially dropped.
            drop_columns: Whether or not to drop the columns that this method
              finds.

        Returns:
            A list of columns that meet the condition.
        """

        # Ensure only one argument is given
        assert (proportion is None and num_missing is not None) or \
                (proportion is not None and num_missing is None)

        # Run the stats getter
        self.get_dataset_stats()

        column_to_index = {}
        index = 0
        # Create map of column names to index
        for column in self.datacleaner.data.columns:
            column_to_index[column] = index
            index += 1

        results = []

        if proportion is not None:
            threshhold = math.floor(self.stats['num_entries'] * proportion)
        else:
            threshhold = num_missing

        for column in self.datacleaner.data.columns:
            if self.stats['empty_cells_per_column'][column_to_index[column]] \
                        >= threshhold:
                results.append(column)

        if drop_columns:
            self.datacleaner.remove_columns(results)

        return results
            

    def get_rows_proportional_missing(self, proportion: float=None, 
            num_missing: int=None, drop_rows: bool=False):
        """Gets rows that have a certain proportion/number of empty cells.
        
        Args:
            proportion: A value in [0, 1] representing the proportion of empty
              cells compared to the total number of columns, that if exceeded,
              should be returned (and potentially dropped).
            num_missing: The number of missing cells that must be in a row
              to be returned, and potentially dropped.
            drop_rows: Whether or not to drop the rows that this method
              finds.

        Returns:
            A list of rows that meet the condition.
        """

        # Ensure only one argument is given
        assert (proportion is None and num_missing is not None) or \
                (proportion is not None and num_missing is None)

        # Run the stats getter
        self.get_dataset_stats()

        results = []

        if proportion is not None:
            threshhold = math.floor(self.stats['num_features'] * proportion)
        else:
            threshhold = num_missing

        for row in self.datacleaner.data.index.values:
            if self.stats['empty_cells_per_row'][row] >= threshhold:
                results.append(row) 

        if drop_rows:
            self.datacleaner.remove_entries(results)

        return results
            

    def k_means(self, k: int, random_state: int=10, 
            max_iter: int=300) -> float:
        """Computes the k means algorithm on the data.
        
        Args:
            k: The number of clusters to assign the data to.
            random_state: Assign an int to ensure reproducible results.
            max_iter: The maximum number of iterations to run when fitting.

        Returns:
            Time taken to compute the algorithm
        """

        # Get start time
        start_time = time.time()

        # Log initial call to begin k means
        self.logger.logger.info("k_means started")
        
        # Perform k means, save results and k
        self.kmeans = KMeans(n_clusters=k, random_state=random_state,
                max_iter=max_iter).fit(self.datacleaner.data)
        self.k = k
        
        # Log success
        self.logger.logger.info("k_means success")

        return time.time() - start_time


    def get_kmeans_summary_string(self) -> str:
        """Returns a string summary of a k means analysis."""

        # Check that there is kmeans to represent
        if self.kmeans is None:
            raise Exception("K means has not been trained yet.")

        self.logger.logger.info("k_means string summary success")
        return str(self.kmeans)


    def predict_from_kmeans(self, data: pd.DataFrame, 
            update_model: bool=True) -> Tuple[list, float]:
        """Predicts cluster classification with an already trained model. 
        
        Args:
            data: Dataframe of entries to predict clusters for.
            update_model: If True, changes the model saved to account for 
              this new data too. 
        
        Returns:
            A tuple, with first element being the cluster index for each entry 
            of the input data and the time taken for the function to compute. 
            Second element is time taken.
        """

        # Get start time
        start_time = time.time()

        # Log beginning of prediction
        self.logger.logger.info("predict from k means start")

        # Check that there is kmeans to plot on
        if self.kmeans is None:
            raise Exception("K means has not been trained yet")

        if update_model:
            classifications = self.kmeans.fit_predict(data)
        else:
            classifications = self.kmeans.predict(data)

        # Log success
        self.logger.logger.info("k_means prediction success")

        return (classifications, time.time() - start_time)

        
    def random_forest(self, target_column: str, subset_X: list=None,
            data_split: float=0.8, random_state: int=10) -> float:
        """ Performs a random forest regression.

        Args:
            target_column: The header for the column to target.
            subset_X: A list of column headers to use only, instead of all of
              the data. None will use all of the data.
            data_split: The proportion of the data to use in the training set.
            random_state: Set to an int for reproducible results.

        Returns:
            The predictions made on the testing set after training.
            The actual values of the testing set.
            Time taken to complete.
        """
        # Get start time
        start_time = time.time()

        # Log beginning of rf analysis
        self.logger.logger.info("rf start")

        # Get the target column and data without the target column
        y = self.datacleaner.data[target_column]

        self.rf_subset_X = subset_X

        # Save this for the importance plotter
        self.rf_target_column = target_column

        if subset_X is None:
            X = self.datacleaner.data.drop(columns=target_column)
            self.rf_num_predictors = self.datacleaner.data.shape[1] - 1
        else:
            X = self.datacleaner.data[subset_X]
            self.rf_num_predictors = len(subset_X)

        train_X, test_X, train_y, test_y = train_test_split(X, y, 
                train_size=data_split, random_state=random_state)

        # Create and fit the random forest
        self.rf = RandomForestRegressor(random_state=random_state)
        self.rf.fit(train_X, train_y)

        test_predictions = self.rf.predict(test_X)
        
        # Save these as class attribute for later visualisations
        self.rf_test_predictions = test_predictions
        self.rf_test_y = test_y

        # Log success
        self.logger.logger.info("rf success")

        return test_predictions, test_y, time.time() - start_time

    
    def predict_random_forest(self, 
            data: pd.DataFrame) -> Tuple[list, float]:
        """Makes a prediction off of data with a pretrained random forest.

        Args:
            data: A dataframe of entries to predict.

        Returns:
            A 2-tuple of predictions, time taken.
        """
        # Get start time
        start_time = time.time()

        # Log start
        self.logger.logger.info("predict from rf start")

        # Check that there is an rf to predict on
        if self.rf is None:
            raise Exception("Random forest has not been trained yet.")

        # Log success
        self.logger.logger.info("predict from rf success")

        return (self.rf.predict(data), time.time() - start_time)


    def svm_regression(self, target_column: str, random_state=10):
        """Uses a support vector machine to perform a regression.
        
        Args:
            target_column: String of the column to predict.
            random_state: An int to ensure results are reproducible.
        """
        pass #TODO


    def dense_nn_regression(self, target_column: str, subset_X: list=None,
            data_split: float=0.8, filename: str=None, hist_filename: str=None,
            epochs: int=5, random_state=10, verbosity: int=2) -> (float):
        """Uses a simple dense neural network to perform a regression.
        
        Args:
            subset_X: The columns to use in the model. If None the model will
              use all column in the data.
            target_column: The label of the column to predict.
            data_split: The proportion of data that should be used for the
              training and testing sets. The training testing split will be
              one quarter of this value.
            filename: None doesn't save model.
            epochs: The number of epochs to train with.
            random_state: Set to int for reproducible data splits.
            verbosity: Set verbosity for tf.keras.Model.fit

        Returns:
            Time taken to complete this method.
        """
        # Log start
        self.logger.logger.info("dense nn regression start")

        # Get start time
        start_time = time.time()

        if subset_X is not None:
            subset_X += [target_column]
            data = self.datacleaner.data[subset_X]
        else:
            data = self.datacleaner.data

        # Split data into training and testing sets
        train_data, test_data = train_test_split(data, train_size=data_split,
                random_state=random_state)

        # Split the target out of each set
        train_X = train_data.copy()
        test_X = test_data.copy()
        train_y = train_X.pop(target_column)
        test_y = test_X.pop(target_column)

        # Create a normalisation layer to normalise the data first
        normaliser = tf.keras.layers.experimental.preprocessing.Normalization()
        normaliser.adapt(np.array(train_X))

        # Define the model
        model = tf.keras.Sequential([
            normaliser,
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(1)
        ])

        # Compile the model with standard inputs
        model.compile(loss='mean_absolute_error', 
                optimizer=tf.keras.optimizers.Adam())

        # Train the model
        self.dense_nn = model.fit(train_X, train_y, validation_split=0.25, 
                epochs=epochs, verbose=verbosity)

        # Log success
        self.logger.logger.info("dense nn regression success")
        
        if filename is not None:
            model.save(filename)

        if hist_filename is not None:
            self.save_history_nn(hist_filename)

        return time.time() - start_time


    def save_history_nn(self, filename: str) -> None:
        """Saves the history variable to a file (named filename)."""
 
        hist_df = pd.DataFrame(self.dense_nn.history) 
        with open(filename, mode='w') as f:
            hist_df.to_csv(f)


    def predict_from_nn(self, data: pd.DataFrame, target_column: str,
            model_path: str) -> Tuple[list, list]:
        """Makes predictions from a neural network."""

        model = tf.keras.models.load_model(model_path)

        X = data.copy()
        y = X.pop(target_column)

        loss = model.evaluate(X, y)
        predictions = model.predict(X)
        return predictions, loss


    def sgd_regression(self, target_column: str):
        """Applies stochastic gradient descent to perform a regression."""
        pass #TODO


    def pca_sk(self, subset_X: list=None, 
            normalise_data: bool=True, n_components=None,
            target_column: str=None, save_transform_name: str=None) -> float:
        """Create and fit a dataset to a PCA.
        
        Args:
            subset_X: A list of column headers to use in model. None will use
              all columns in the dataframe.
            normalise_data: Whether or not to normalise the data first.
            n_components: As in sickit, (0, 1) will give a proportion of
              variance explained, < n (n features) will give that number of
              principle components.
            target_column: If None, PCA will be done on every column. If
              a column is specified, then that column will not be included
              in the PCA. Note if subset_X is specified, this arg will not be
              used, and the PCA will be performed on subset_X.
            save_transform_name: If the transformed data is to be saved, this
              arg specifies the filename.

        Returns:
            Time taken to complete this method.
        """

        # Get start time
        start_time = time.time()

        # Log start
        self.logger.logger.info("pca start")

        # Work with a subset or the entire dataset
        if subset_X is None:
            
            # Drop the target column if necessary
            if target_column is not None:
                data = self.datacleaner.data.drop(target_column)
            else:
                data = self.datacleaner.data

            # Creates a copy with a scaled version
            if normalise_data:
                scaled = MinMaxScaler().fit_transform(data)
                self.pca = PCA(n_components=n_components).fit(
                        pd.DataFrame(scaled))
            else:
                self.pca = PCA(n_components=n_components).fit(
                        self.datacleaner.data)
        else:
            if normalise_data:
                scaled = MinMaxScaler().fit_transform(
                        self.datacleaner.data[subset_X])
                self.pca = PCA(n_components=n_components).fit(
                        pd.DataFrame(scaled))
            else:
                self.pca = PCA(n_components=n_components).fit(
                        self.datacleaner.data[subset_X])

        # Log success
        self.logger.logger.info("pca success")

        # Save transformed data if necessary
        if save_transform_name is not None:
            transformed_data = self.pca.transform(data)
            transformed_data.to_csv(save_transform_name)

        return (time.time() - start_time)


    def functional_analysis(self, normalise_data: bool=True, 
            random_state: int=10) -> pd.DataFrame:
        """Performs a factor analysis on the data.
        
        Args:
            normalise_data: Whether or not to normalise the data.
            random_state: An int to ensure results are reproducible.

        Returns:
            The dataframe transformed into the latent space.    
        """
        # TODO Might need similar adjustments as with PCA
        # Log start
        self.logger.logger.info("fa start")

        if normalise_data:
            scaled = MinMaxScaler().fit_transform(self.datacleaner.data)
            self.fa = FactorAnalysis(random_state=random_state)

        new_data = self.fa.fit_transform(scaled)

        self.logger.logger.info("fa success")

        return new_data


    def hierarchical_clustering(self, transpose: bool=False,
            normalise: bool=True, subset_X: list=None) -> float:
        """Performs a hierarchical clustering on the data.
        
        Args:
            transpose: If True, transpose the data first.
            normalise: If True, normalise the data first.
            subset_X: Specifies the columns to use in the algorithm. None will
              use all columns in the dataframe.

        Return:
            Time taken to complete method.
        """

        # Get start time
        start_time = time.time()

        # Log start
        self.logger.logger.info("hca start")

        if subset_X is not None:
            data = self.datacleaner.data[subset_X]
        else:
            data = self.datacleaner.data

        # Normalise if necessary
        if normalise:
            data = MinMaxScaler().fit_transform(data)
        
        # Transpose if necessary
        data = data.T if transpose else data

        self.hca = linkage(data, method='ward')

        # Log success
        self.logger.logger.info("hca success")

        return time.time() - start_time


    def correlation_coefficients(self, subset_X: list=None) -> float:
        """Calculates various correlation coefficients for the data.
        
        Args:
            subset_X: Specifies the columns to use in the algorithm. None will
              use all columns in the dataframe.
        
        Returns:
            Time taken to complete method.
        """

        # Get start time
        start_time = time.time()

        # Log start
        self.logger.logger.info("cc start")

        if subset_X is None:
            data = self.datacleaner.data
            self.cc_subset_X = self.datacleaner.data.columns
        else:
            data = self.datacleaner.data[subset_X]
            self.cc_subset_X = subset_X
        
        pearson = data.corr(method='pearson')
        kendall = data.corr(method="kendall")
        spearman = data.corr(method="spearman")

        self.correlations = {
            'pearson': pearson,
            'kendall': kendall,
            'spearman': spearman
        }

        # Log success
        self.logger.logger.info("cc success")
        
        return time.time() - start_time


    def ada_boost(self, target_column: str, subset_X: list=None,
            random_state: int=10, data_split: float=0.8) -> float:
        """Creates and fits an AdaBoost regressor.

        Args:
            target_column: The name of the column to target.
            subset_X: A list of columns to use as predictors. None will use
              the entire dataset with the target column dropped.
            random_state: Set the random state in sklearn.

        Returns:
            Predictions made on the testing set.
            The actual values of the testing set.
            Time taken to complete function.
        """

        # Start time
        start_time = time.time()

        self.adaboost = AdaBoostRegressor(random_state=random_state)

        # Get the target column and data without the target column
        y = self.datacleaner.data[target_column]

        self.adaboost_subset_X = subset_X

        # Save this for the importance plotter
        self.ada_target_column = target_column

        # Get Data
        if subset_X is None:
            X = self.datacleaner.data.drop(columns=target_column)
            self.adaboost_num_predictors = self.datacleaner.data.shape[1] - 1
        else:
            X = self.datacleaner.data[subset_X]
            self.adaboost_num_predictors = len(subset_X)

        train_X, test_X, train_y, test_y = train_test_split(X, y, 
                train_size=data_split, random_state=random_state)


        # Fit the training data        
        self.adaboost.fit(X=train_X, y=train_y)

        test_predictions = self.adaboost.predict(test_X)

        # Save these as class attribute for later visualisations
        self.ada_test_predictions = test_predictions
        self.ada_test_y = test_y

        # Log success
        self.logger.logger.info("ada success")

        return test_predictions, test_y, time.time() - start_time


    def compare_ada_rf_xg(self, top_n: int=10) -> list:
        """Returns a list of common important features from rf, xg and AdaBoost.

        Specifically, it returns the common elements in the top n features from
        this Analyser's rf and adaboost models.

        Args:
            top_n: The top n features to search in the models.

        Returns:
            A list of common features from the models.

        Raises:
            Exception: Both rf and adaboost must be not None.
        """

        if self.rf is None or self.adaboost is None or self.xgboost is None:
            raise Exception(
                    "RF or XGBoost or AdaBoost has not been trained yet.")

        rf_importances = np.argsort(
                self.rf.feature_importances_)[::-1][0:top_n]
        ada_importances = np.argsort(
                self.adaboost.feature_importances_)[::-1][0:top_n]
        xg_importances = np.argsort(
                self.xgboost.feature_importances_)[::-1][0:top_n]


        if self.rf_subset_X is None:
            labels_rf = [
                    self.datacleaner.data.columns[i] for i in rf_importances]
        else:
            labels_rf = [
                    self.rf_subset_X[i] for i in rf_importances]

        if self.adaboost_subset_X is None:
            labels_ada = [
                    self.datacleaner.data.columns[i] for i in ada_importances]
        else:
            labels_ada = [
                    self.adaboost_subset_X[i] for i in ada_importances]

        if self.xgboost_subset_X is None:
            labels_xg = [
                    self.datacleaner.data.columns[i] for i in xg_importances]
        else:
            labels_xg = [
                    self.xgboost_subset_X[i] for i in xg_importances]

        intersect = np.intersect1d(labels_rf, labels_ada)
        intersect = np.intersect1d(intersect, labels_xg)
        return intersect


    def xg_boost(self, target_column: str, subset_X: list=None,
            random_state: int=10, data_split: float=0.8):
        """Trains and tests an XG Boost decision tree on the data.
        
        Args:
            target_column: The name of the column to target.
            subset_X: A list of columns to use as predictors. None will use
              the entire dataset with the target column dropped.
            random_state: Set the random state in sklearn.

        Returns:
            Predictions made on the testing set.
            The actual values of the testing set.
            Time taken to complete function.
        """

        start_time = time.time()

        self.xgboost = xgb.XGBRegressor(random_state=random_state)

        # Get the target column and data without the target column
        y = self.datacleaner.data[target_column]

        self.xgboost_subset_X = subset_X

        # Save this for the importance plotter
        self.xgboost_target_column = target_column

        # Get Data
        if subset_X is None:
            X = self.datacleaner.data.drop(columns=target_column)
            self.xgboost_num_predictors = self.datacleaner.data.shape[1] - 1
        else:
            X = self.datacleaner.data[subset_X]
            self.xgboost_num_predictors = len(subset_X)

        train_X, test_X, train_y, test_y = train_test_split(X, y, 
                train_size=data_split, random_state=random_state)


        # Fit the training data        
        self.xgboost.fit(X=train_X, y=train_y)

        test_predictions = self.xgboost.predict(test_X)

        # Save these as class attribute for later visualisations
        self.xgboost_test_predictions = test_predictions
        self.xgboost_test_y = test_y

        # Log success
        self.logger.logger.info("xg success")

        return test_predictions, test_y, time.time() - start_time

        