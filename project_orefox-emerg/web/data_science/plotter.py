"""Contains a class that handles visualisations of analysis."""

import sys
from typing import Tuple
import io
import gzip

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import set_link_color_palette
import tensorflow as tf



# Define some colours for the plot themes
COPPER = '#C77C53'
COPPER2 = '#170E09' # TODO Make a theme class


class Plotter:
    """
    Handles visualisation creation directly of the data in a datacleaner,
    and of the analysis done in the analyser.

    Attributes:
        analyser: The Analyser that will be interpreted.
        logger: The logger to record events.
    """

    def __init__(self, analyser):
        """Create a plotter"""
        self.analyser = analyser

        # Get the logger as a member of this class to make logging calls
        # less complicated
        self.logger = analyser.logger

        self.logger.logger.info('Created Plotter')


    def visualise_empty_cells(self, filename: str, 
            figsize: Tuple[int,int]=(10,10), label_font_size: int=16,
            title_font_size: int=24, rotation: int=0, colour: str=COPPER):
        """Creates a scatter plot visualising empty cells.
        
        Args:
            filename: The filename to save the plot to.
            figsize: The size in which to create the figure.
            label_font_size: The font size of axis labels.
            title_font_size: The font size of title.
            rotation: The rotation of the tick labels.
            colour: The colour of the bars in the plot.
        """

        # Check stats have been called
        if self.analyser.stats is None:
            raise Exception('Stats have not been calculated yet.')

        # Plot each column
        plt.figure(figsize=figsize)
        for index in range(len(self.analyser.datacleaner.data.columns)):
            plt.scatter(index, 
                    self.analyser.stats['empty_cells_per_column'][index],
                    color=colour)

        names = self.analyser.datacleaner.data.columns

        plt.title("Missing Values", fontsize=title_font_size)
        
        plt.xticks(range(len(self.analyser.datacleaner.data.columns)), 
                labels=names, rotation=rotation, fontsize=label_font_size)

        plt.savefig(filename)
        plt.close()


    def visualise_empty_cells_bar(self, filename: str, 
            figsize: Tuple[int,int]=(10,10), label_font_size: int=16,
            title_font_size: int=24, rotation: int=0, colour: str=COPPER):
        """Creates a bar graph visualising empty cells.
        
        Args:
            filename: The filename to save the plot to.
            figsize: The size in which to create the figure.
            label_font_size: The font size of axis labels.
            title_font_size: The font size of title.
            rotation: The rotation of the tick labels.
            colour: The colour of the bars in the plot.
        """

        # Check stats have been called
        if self.analyser.stats is None:
            raise Exception('Stats have not been calculated yet.')

        # Plot each column
        plt.figure(figsize=figsize)
        
        plt.bar(range(len(
                    self.analyser.stats['empty_cells_per_column'].keys())), 
                height=self.analyser.stats['empty_cells_per_column'].values(),
                color=colour)

        names = self.analyser.datacleaner.data.columns

        plt.title("Missing Values", fontsize=title_font_size)
        
        plt.xticks(range(len(self.analyser.datacleaner.data.columns)), 
                labels=names, rotation=rotation, fontsize=label_font_size)

        plt.savefig(filename)
        plt.close()


    def plot_2d_comparison_kmeans(self, col1: str, col2: str, filename: str,
            plot_centres: bool=True, 
            colours: list=['k', 'r', 'b', 'c']) -> None:
        """Plots a two dimensional slice of the clustered data.

        Args:
            col1: The header of the first dimension to plot
            col2: The header of the second dimension to plot
            filename: The name of the file to save the plot to
            plot_centres: If True, also plots the 2D slice of the cluster
              centre.
            colours: The list of colours for different clusters to use.
        """
        # Log start
        self.logger.logger.info("2d kmeans plot start")

        # Check that there is kmeans to plot on
        if self.analyser.kmeans is None:
            raise Exception("K means has not been trained yet.")

        # Get data for the two dimensions being compared
        data_x = self.analyser.datacleaner.data[col1]
        data_y = self.analyser.datacleaner.data[col2]

        cluster_colours = [colours[i] for i in self.analyser.kmeans.labels_]

        # Plot them against each other
        plt.scatter(data_x, data_y, c=cluster_colours)

        cluster_index_x = self.analyser.datacleaner.data.columns.get_loc(col1)
        cluster_index_y = self.analyser.datacleaner.data.columns.get_loc(col2)

        # Plot the centres
        if plot_centres:
            for i in range(self.analyser.k):
                cluster_x = self.analyser.kmeans.cluster_centers_[i][
                    cluster_index_x]
                cluster_y = self.analyser.kmeans.cluster_centers_[i][
                    cluster_index_y]

                plt.scatter(cluster_x, cluster_y, c=colours[i], marker='*')

        # Set the labels for the axes
        plt.xlabel('{} ({})'.format(col1, self.analyser.datacleaner.units))
        plt.ylabel('{} ({})'.format(col2, self.analyser.datacleaner.units))

        # Save the image
        plt.savefig(filename)

        plt.close()

        # Log success
        self.logger.logger.info("2d kmeans plot success")


    def plot_rf_importances(self, filename: str, ordered: bool=True, 
            big_to_small: bool=True, horizontal: bool=True, 
            figsize: Tuple[int,int]=(10,10), label_font_size: int=16,
            title_font_size: int=24, rotation: int=0, 
            colour: str=COPPER) -> None:
        """
        Plots the importance of each feature in a pre-run random forest. 
        
        Args:
            filename: The filename to save the plot to.
            ordered: Whether or not to sort the importances in ascending order
            big_to_small: If ordered is True, this indicates whether the plot
              should be in ascending or descending order.
            horizontal: If True, plots a horizontal bar graph. Otherwise,
              a normal bar graph (vertical) is created
            figsize: The size in which to create the figure.
            label_font_size: The font size of axis labels.
            title_font_size: The font size of title.
            rotation: The rotation of the tick labels.
            colour: The colour of the bars in the plot.
        """

        # Log start
        self.logger.logger.info("rf importances plot start")

        # Check that there is an rf to get the importances of, also covers the
        # target column set.
        if self.analyser.rf is None:
            raise Exception("Random forest has not been trained yet.")
        
        # Get the importances
        importances = self.analyser.rf.feature_importances_

        # Sort if necessary
        if ordered:
            if big_to_small:
                index_sorted = np.argsort(importances)[::-1]
                importances = np.sort(importances)[::-1]

            else:
                index_sorted = np.argsort(importances)
                importances = np.sort(importances)
        else:
            index_sorted = [i for i in range(len(importances))]
            
        if self.analyser.rf_subset_X is None:
            labels = [self.analyser.datacleaner.data.columns[i] 
                    for i in index_sorted]
        else:
            labels = [self.analyser.rf_subset_X[i] for i in index_sorted]

        # Create figure and initialise with figure size
        plt.figure(figsize=figsize)

        if horizontal:
            # Plot the importances
            plt.barh(range(self.analyser.rf_num_predictors), 
                    importances, color=colour)

            # Set labels
            plt.yticks(ticks=range(self.analyser.rf_num_predictors), 
                    labels=labels, rotation=rotation, fontsize=label_font_size)

            plt.title("Importances Random Forest Predicting {}".format(
                    self.analyser.rf_target_column), fontsize=title_font_size)

        else:
            # Plot the importances
            plt.bar(range(self.analyser.rf_num_predictors), importances, 
                    color=colour)

            # Set labels
            plt.xticks(range(self.analyser.rf_num_predictors), labels,
                    rotation=rotation, fontsize=label_font_size)

            plt.title("Importances Random Forest Predicting {}".format(
                    self.analyser.rf_target_column), fontsize=title_font_size)


        # Save the figure
        plt.savefig(filename)

        plt.close()
        self.logger.logger.info("rf importances plot success")


    def plot_pca_feature_bar(self, filename: str, 
            figsize: Tuple[int,int]=(10,10), label_font_size: int=16, 
            title_font_size: int=24, rotation: int=0, 
            colour: str=COPPER) -> None:
        """Plots features with their importance from a PCA.
        
        Args:
            filename: The filename to save the plot to.
            figsize: The size in which to create the figure.
            label_font_size: The font size of axis labels.
            title_font_size: The font size of title.
            rotation: The rotation of the tick labels.
            colour: The colour of the bars in the plot.
        """
        # Log start
        self.logger.logger.info("pca feature bar start")

        # Check there is a PCA
        if self.analyser.pca is None:
            raise Exception("PCA has not been trained yet.")
        
        # Get the explained variance ratio from the PCA
        explained_var_ratio = self.analyser.pca.explained_variance_ratio_

        # Create figure and initialise with figure size
        plt.figure(figsize=figsize)

        # Plot the bar graph
        plt.barh(range(len(explained_var_ratio)), np.flip(explained_var_ratio), 
                color=colour)

        plt.yticks(ticks=range(len(explained_var_ratio)), 
                rotation=rotation, fontsize=label_font_size)

        plt.title("Explained Variance of Total Data per Feature", 
                fontsize=title_font_size)

        # Save and close the figure
        plt.savefig(filename)
        plt.close()

        # Log success
        self.logger.logger.info("pca feature bar success")


    def plot_pca_cumulative_importance(self, filename: str, 
            figsize: Tuple[int,int]=(10,10), label_font_size: int=16, 
            title_font_size: int=24, rotation: int=0,
            colour: str=COPPER) -> None:
        """Plots the cumulative explained variance of the features.
        
        Args:
            filename: The filename to save the plot to.
            figsize: The size in which to create the figure.
            label_font_size: The font size of axis labels.
            title_font_size: The font size of title.
            rotation: The rotation of the tick labels.
            colour: The colour of the bars in the plot.
        """
        # Log start
        self.logger.logger.info("pca cumulative bar start")

        # Check there is a PCA
        if self.analyser.pca is None:
            raise Exception("PCA has not been trained yet.")

        # Create figure and initialise with figure size
        plt.figure(figsize=figsize)
        
        # Get the explained variance ratio and cumulative sum
        explained_var_ratio = self.analyser.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_var_ratio)

        # Plot the figure
        plt.plot(range(len(cumulative_variance)), cumulative_variance, 
                color=colour)
        plt.title('Cumulative sum of explained variance for each feature.',
                fontsize=title_font_size)
        plt.xlabel('Principle Component', fontsize=label_font_size)
        plt.ylabel('Explained Variance', fontsize=label_font_size)
        plt.savefig(filename)
        plt.close()

        # Log success
        self.logger.logger.info("pca cumulative bar success")


    def plot_dendrogram(self, filename: str, labels: list, 
            figsize: Tuple[int,int]=(10,10), label_font_size: int=16, 
            title_font_size: int=24, rotation: int=0, 
            colour: list=['b', COPPER, COPPER2, 'k']) -> None:
        """Plots a dendrogram for a trained HCA.
        
        Args:
            filename: The filename to save the plot to.
            labels: A list of strings corresponding to the axis labels.
            figsize: The size in which to create the figure.
            label_font_size: The font size of axis labels.
            title_font_size: The font size of title.
            rotation: The rotation of the tick labels.
            colour: The colour the different cluster thresholds.
        """
        # Log start
        self.logger.logger.info("dendrogram start")

        # Check HCA exists
        if self.analyser.hca is None:
            raise Exception("HCA has not been trained yet.")
        
        # Plot the dendrogram
        plt.figure(figsize=figsize)
        set_link_color_palette(colour)
        dendrogram(self.analyser.hca, labels=labels, leaf_rotation=rotation,
                leaf_font_size=label_font_size, 
                above_threshold_color=colour[-1])

        plt.title("Dendrogram of Hierarchical Clustering", 
                fontsize=title_font_size)
        plt.savefig(filename)
        plt.close()

        # Log success
        self.logger.logger.info("dendrogram success")


    def plot_correlations_heatmap(self, filenames: list, 
            figsize: Tuple[int,int]=(10,10), cmap: str='copper_r') -> None:
        """Creates heatmaps for correlation coefficients of data.

        Args:
            filenames: The list of files to save the heatmaps to.
            figsize: The size in which to create the figure.
            cmap: The colour map to use on the heatmaps.
        """
        # Log start
        self.logger.logger.info("cc heatmap start")
        data = []
        counter = 0
        for correlation in self.analyser.correlations:
            plt.figure(figsize=figsize)
            plt.imshow(self.analyser.correlations[correlation], 
                    cmap=cmap, interpolation=None)
            plt.colorbar()
            plt.xticks(range(len(self.analyser.cc_subset_X)), 
                    self.analyser.cc_subset_X, rotation=90)
            plt.yticks(range(len(self.analyser.cc_subset_X)), 
                    self.analyser.cc_subset_X)
            plt.title('{}'.format(correlation))
            

            filename = filenames[counter]
            content = io.BytesIO()
            format = filename.split('.')[-1]
            plt.savefig(content, format=format)
            data.append({
                'filename': filename,
                'expected_filename': filename[:-(len(filename.split('.')[-1])+1)], # file name without extension
                'content': content,
            })
            plt.close()

            counter += 1

        self.logger.logger.info("cc heatmap success")

        return data


    def plot_tf_model_losses(self, hist_path: str, filename: str):
        """Plots the training and validation loss of a neural network.

        Args:
            model_path: The path of where the saved model is.
            filename: The file to save the plot to.
        """
        model = pd.read_csv(hist_path)
        plt.plot(model['loss'], label='loss')
        plt.plot(model['val_loss'], label='validational loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()


    def plot_model_predictions(self, predictions: np.array, 
            actual: np.array, filename: str) -> None:
        """Plots predictions against their true values in scatterplot.
        
        Args:
            predictions: An array of predictions from tf.predict.
            actual: An array of true values, must be the same length as
              predictions (and correspond index to index).
            filename: File to save the visualisation to.
        """
        plt.scatter(predictions, actual)
        plt.xlabel('True Values')
        plt.ylabel('Predictions')

        x = np.linspace(np.amin(actual), np.amax(actual), 1000)
        plt.plot(x, x, color='k')

        plt.savefig(filename)
        plt.close()


    def bar_tf_model_predictions(self, predictions: np.array,
            actual: np.array, filename: str) -> None:
        """Takes a model and produces a histogram of the errors.
        
        Args:
            predictions: An array of predictions from tf.predict.
            actual: An array of true values, must be the same length as
              predictions (and correspond index to index).
            filename: File to save the visualisation to.
        """
        error = predictions - actual.values.reshape(len(actual), 1)
        plt.hist(error)
        plt.savefig(filename)
        plt.close()


    def plot_adaboost_importances(self, filename: str, ordered: bool=True, 
            big_to_small: bool=True, horizontal: bool=True, 
            figsize: Tuple[int,int]=(10,10), label_font_size: int=16,
            title_font_size: int=24, rotation: int=0, 
            colour: str=COPPER) -> None:
        """
        Plots the importance of each feature in a pre-run AdaBoost. 
        
        Args:
            filename: The filename to save the plot to.
            ordered: Whether or not to sort the importances in ascending order
            big_to_small: If ordered is True, this indicates whether the plot
              should be in ascending or descending order.
            horizontal: If True, plots a horizontal bar graph. Otherwise,
              a normal bar graph (vertical) is created
            figsize: The size in which to create the figure.
            label_font_size: The font size of axis labels.
            title_font_size: The font size of title.
            rotation: The rotation of the tick labels.
            colour: The colour of the bars in the plot.
        """

        # Log start
        self.logger.logger.info("AdaBoost importances plot start")

        # Check that there is an adaboost to get the importances of
        if self.analyser.adaboost is None:
            raise Exception("AdaBoost has not been trained yet.")
        
        # Get the importances
        importances = self.analyser.adaboost.feature_importances_

        # Sort if necessary
        if ordered:
            if big_to_small:
                index_sorted = np.argsort(importances)[::-1]
                importances = np.sort(importances)[::-1]

            else:
                index_sorted = np.argsort(importances)
                importances = np.sort(importances)
        else:
            index_sorted = [i for i in range(len(importances))]
            
        if self.analyser.adaboost_subset_X is None:
            labels = [self.analyser.datacleaner.data.columns[i] 
                    for i in index_sorted]
        else:
            labels = [self.analyser.adaboost_subset_X[i] for i in index_sorted]

        # Create figure and initialise with figure size
        plt.figure(figsize=figsize)

        if horizontal:
            # Plot the importances
            plt.barh(range(self.analyser.adaboost_num_predictors), 
                    importances, color=colour)

            # Set labels
            plt.yticks(ticks=range(self.analyser.adaboost_num_predictors), 
                    labels=labels, rotation=rotation, fontsize=label_font_size)

            plt.title("Importances AdaBoost Predicting {}".format(
                    self.analyser.ada_target_column), fontsize=title_font_size)

        else:
            # Plot the importances
            plt.bar(range(self.analyser.adaboost_num_predictors), importances, 
                    color=colour)

            # Set labels
            plt.xticks(range(self.analyser.adaboost_num_predictors), labels,
                    rotation=rotation, fontsize=label_font_size)

            plt.title("Importances AdaBoost Predicting {}".format(
                    self.analyser.ada_target_column), fontsize=title_font_size)


        # Save the figure
        plt.savefig(filename)

        plt.close()
        self.logger.logger.info("ada importances plot success")


    def plot_xgboost_importances(self, filename: str, ordered: bool=True, 
            big_to_small: bool=True, horizontal: bool=True, 
            figsize: Tuple[int,int]=(10,10), label_font_size: int=16,
            title_font_size: int=24, rotation: int=0, 
            colour: str=COPPER) -> None:
        """
        Plots the importance of each feature in a pre-run XGBoost. 
        
        Args:
            filename: The filename to save the plot to.
            ordered: Whether or not to sort the importances in ascending order
            big_to_small: If ordered is True, this indicates whether the plot
              should be in ascending or descending order.
            horizontal: If True, plots a horizontal bar graph. Otherwise,
              a normal bar graph (vertical) is created
            figsize: The size in which to create the figure.
            label_font_size: The font size of axis labels.
            title_font_size: The font size of title.
            rotation: The rotation of the tick labels.
            colour: The colour of the bars in the plot.
        """

        # Log start
        self.logger.logger.info("XGBoost importances plot start")

        # Check that there is an xgboost to get the importances of
        if self.analyser.xgboost is None:
            raise Exception("XGBoost has not been trained yet.")
        
        # Get the importances
        importances = self.analyser.xgboost.feature_importances_

        # Sort if necessary
        if ordered:
            if big_to_small:
                index_sorted = np.argsort(importances)[::-1]
                importances = np.sort(importances)[::-1]

            else:
                index_sorted = np.argsort(importances)
                importances = np.sort(importances)
        else:
            index_sorted = [i for i in range(len(importances))]
            
        if self.analyser.xgboost_subset_X is None:
            labels = [self.analyser.datacleaner.data.columns[i] 
                    for i in index_sorted]
        else:
            labels = [self.analyser.xgboost_subset_X[i] for i in index_sorted]

        # Create figure and initialise with figure size
        plt.figure(figsize=figsize)

        if horizontal:
            # Plot the importances
            plt.barh(range(self.analyser.xgboost_num_predictors), 
                    importances, color=colour)

            # Set labels
            plt.yticks(ticks=range(self.analyser.xgboost_num_predictors), 
                    labels=labels, rotation=rotation, fontsize=label_font_size)

            plt.title("Importances XGBoost Predicting {}".format(
                    self.analyser.xgboost_target_column), 
                    fontsize=title_font_size)

        else:
            # Plot the importances
            plt.bar(range(self.analyser.xgboost_num_predictors), importances, 
                    color=colour)

            # Set labels
            plt.xticks(range(self.analyser.xgboost_num_predictors), labels,
                    rotation=rotation, fontsize=label_font_size)

            plt.title("Importances XGBoost Predicting {}".format(
                    self.analyser.xgboost_target_column), 
                    fontsize=title_font_size)


        # Save the figure
        plt.savefig(filename)

        plt.close()
        self.logger.logger.info("xg importances plot success")