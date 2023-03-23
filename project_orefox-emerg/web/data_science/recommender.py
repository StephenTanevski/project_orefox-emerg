"""Holds a recommendation engine."""


import numpy as np

from . import analyser


class Recommender:

    @staticmethod
    def recommend_cleaner(datacleaner, missing_tol=0.75):
        """Collects and runs checks on a datacleaner to make recommendations.
        
        Args:
            datacleaner: The datacleaner to base recommendations off of.
            missing_tol: The tolerance where if there is a greater proportion
              of missing data in the column or row, it will be added to the
              lists returned.

        Returns:
            constant_columns: A list of columns with one unique values.
            missing_rows: A list of rows with missing_tol amounts of missing
              data.
            missing_cols: A list of columns with missing_tol amounts of
              missing data.
        """

        # Get stats of data
        a = analyser.Analyser(datacleaner)
        a.get_dataset_stats()

        # See if any columns are constant
        constant_columns = []
        for column in datacleaner.data.columns:
            if len(np.unique(datacleaner.data[column])) == 1:
                constant_columns.append(column)

        # Get rows and columns that have over missing_tol missing entries
        missing_rows = a.get_rows_proportional_missing(proportion=missing_tol)
        missing_cols = a.get_columns_proportional_missing(
                proportion=missing_tol)

        return constant_columns, missing_rows, missing_cols

        
    @staticmethod
    def recommend_analysis(analyser):
        """Makes recommendations on an analyser."""
        # Currently there isn't all that much to do here
        # Eventually some of the data can be used to help here though
        pass


    @staticmethod
    def recommend_plot(plotter):
        """Makes recommendations on an analyser."""
        # Currently there isn't all that much to do here
        pass