from datetime import date
import os

import pandas as pd
import numpy as np
#from sklearn import preprocessing
import io
import gzip


class DataCleaner:
    """Holds methods for importing and manipulating data."""

    def __init__(self, filename, testing=False, lab=None, ppb=True, 
            index_col=None, null_replace=None):
        """
        Initially loads the data from 'filename'. If testing is True, this
        only loads a small subset (when a lab is not specified). The lab
        can also be specified to make the data more immediately usable.

        Currently supports ALS and OSNACA.
        """

        # Check file exists
        if not os.path.isfile(filename):
            raise(FileNotFoundError)

        # Initialise some meta-variables for a general dataset
        self.title = filename
        self.client = None
        self.num_samples = None
        self.date_received = None
        self.date_finalised = None
        self.project = None
        self.certificate_comments = None
        self.PO_Number = None
        self.units = None

        # Save the lab name, 'Generic' is used when not specified
        self.lab = lab if lab is not None else 'Generic'

        # Initially load data

        # Case where there is no lab specified
        if lab is None:
            if testing:
                # Only use a small subset of entries for performance
                self.data = pd.read_csv(filename, nrows=9000, 
                        index_col=index_col)
            else:
                # TODO Handle low memory case
                self.data = pd.read_csv(filename, index_col=index_col)
                
        # Data from ALS
        elif lab == 'ALS':
            # Get the meta-variables listed at the beginning of the file
            meta_data = pd.read_csv(filename, nrows=7, usecols=[0], header=None)
            self.set_meta_variables(meta_data)
            
            # Read in the data
            self.data = pd.read_csv(
                filename,
                index_col=0,
                skiprows=7,
                header=1
            )

            # Add method to headers to reduce the header to one row
            self.update_headers_ALS(filename)

            # Deal with cells that have < or > in them
            self.handle_inequalities_ALS()

            # Convert to same units
            self.units = 'ppb' if ppb else 'ppm'
            self.convert_uniform_units('DESCRIPTION', ppb=ppb)

            self.set_data_type(self.data.columns, np.float32)

        # Data from OSNACA
        elif lab == 'OSNACA':
            self.data = pd.read_excel(
                filename,
                index_col=1,
                na_values=[' ']
            )

            # Extract descriptions at bottom of file
            self.descriptions = self.get_descriptions_OSNACA(filename)

            # Remove descriptions column from data
            self.data = self.data.drop(['Batch_no'], axis='columns')

            # Add method to headers
            self.update_headers_OSNACA(filename)

            # No need to convert units because they're all in ppm
            self.units = 'ppm' # Last unit in OSNACA is g to measure weight
            self.data.drop(['UNITS'], axis=0, inplace=True)
            
            # Replace empty cells with a value
            if null_replace is not None:
                self.data = self.data.fillna(value=null_replace)

            self.set_data_type(self.data.columns, np.float32)


    def __repr__(self):
        # TODO Tidy up
        result = ""
        result += self.title + '\n'
        result += self.client + '\n'
        result += str(self.num_samples) + '\n'
        result += str(self.date_received) + '\n'
        result += str(self.date_finalised) + '\n'
        result += self.project + '\n'
        result += self.certificate_comments + '\n'
        result += self.PO_Number
        return result


    def set_meta_variables(self, meta_data):
        """
        Method to handle initialisation of meta-variables. Currently only
        useful for ALS because these are specified in the document.
        """
        self.title = meta_data[0][0]
        self.client = meta_data[0][1][9:]
        self.num_samples = int(meta_data[0][2][15:])

        # Get string of dates
        rec_str = meta_data[0][3][16:26]
        rec_str = rec_str.split('-')
        self.date_received = date(
            int(rec_str[0]),
            int(rec_str[1]),
            int(rec_str[2])
        )

        # Get the dates and save them as date objects
        final_str = meta_data[0][3][45:]
        final_str = final_str.split('-')
        self.date_finalised= date(
            int(final_str[0]),
            int(final_str[1]),
            int(final_str[2])
        )

        self.project = meta_data[0][4][10:]
        self.certificate_comments = meta_data[0][5][23:]
        self.PO_Number = meta_data[0][6][12:]


    def update_headers_ALS(self, filename):
        """ 
        Helper function that takes data from ALS and appends the method of
        measurement to the end of the element, so that the header of each
        column is one cell.
        """

        # Read in file to just get the method name
        methods = pd.read_csv(filename, skiprows=7, nrows=0, header=0)
        methods = list(methods)
            
        # Assume method name does not have . in it
        for method_index in range(len(methods)):
            # Skip the top left one because it labels the samples
            if method_index == 0:
                continue
                
            # Remove the number pandas puts at the end for uniqueness
            methods[method_index] = methods[method_index].split('.')[0]

        # Update the actual data to have in form Element-Method
        self.data.columns = [self.data.columns[column] + ': ' 
                + methods[column + 1] 
                    for column in range(len(self.data.columns))]


    def update_headers_OSNACA(self, filename):
        """
        Similar to update_headers_ALS except specifically defined for the
        OSNACA data.
        """
        # Read in file to just get the method name
        methods = pd.read_excel(filename, skiprows=2, nrows=0, header=0)
        methods = list(methods)
            
        # Assume method name does not have . in it
        for method_index in range(len(methods)):
            # Skip the two top left one because it labels the samples
            # and description
            if method_index == 0 or method_index == 1:
                continue
                
            # Remove the number pandas puts at the end for uniqueness
            methods[method_index] = methods[method_index].split('.')[0]

        # Update the actual data to have in form Element-Method
        self.data.columns = [self.data.columns[column] + ': ' 
                + methods[column + 2] 
                    for column in range(len(self.data.columns))]

        self.data.drop(['METHOD'], axis=0, inplace=True)

        det_lims = pd.read_excel(filename, skiprows=3, nrows=0, header=0)
        det_lims = list(det_lims)

        # Go through each index
        for lim_index in range(len(det_lims)):

            # Skip the first two again
            if lim_index in [0, 1]:
                continue
            
            # Split string
            split = str(det_lims[lim_index]).split('.')
            
            # Case where number is in the form 0.xx
            # This assumes numbers are either integers or in (0, 1)
            if split[0][0] == '0':
                det_lims[lim_index] = split[0] + '.' + split[1]
            else:
                det_lims[lim_index] = '' + split[0]


        # Get what the new column names will be
        new_names = []    
        for column in range(len(self.data.columns)):
            new_names.append(self.data.columns[column] 
                    + ', D.L.: ' 
                    + det_lims[column + 2]
            )

        # Update names and drop detection limit row
        self.data.columns = new_names
        self.data.drop('DETECTION LIMIT', axis=0, inplace=True)


    def handle_inequalities_ALS(self):
        """
        Replaces the cells that involve inequalities rather than specific
        measurements. Currently sets < inequalities to be 0 and > to be equal
        to the threshold.

        TODO Needs better or at least some more ways of handling these symbols
        """
        # Most likely not the most efficient way to find all < cells
        for row in range(self.data.shape[0]):
            for column in range(self.data.shape[1]):
                if self.data.iloc[row][column][0] == '<':
                    self.data.iloc[row][column] = 0

                elif self.data.iloc[row][column][0] == '>':
                    self.data.iloc[row][column] = \
                            self.data.iloc[row][column][1:]


    def convert_uniform_units(self, unit_row_header, ppb=True):
        """
        This converts the values in a dataset to all be the same. Assumes that
        all values are either ppm, ppb or something that does not need to be
        converted because it is already uniform.

        unit_row_header is the name of the row that holds the unit description.
        """

        # Get the units of each column
        units = self.data.loc[unit_row_header]

        # Get which indexes are in ppm specifically
        column_index = 1
        unit_indexes = []
        for unit in units:
            if ppb:
                if unit == 'ppm':
                    unit_indexes.append(column_index)
            else:
                if unit == 'ppb':
                    unit_indexes.append(column_index)
            column_index += 1

        # Get rid of units row
        self.data.drop(unit_row_header, inplace=True)
        
        # Cast data (currently string) to float
        self.data = self.data.astype(np.float32)

        # Convert ppm to ppb
        for index in unit_indexes:
            if ppb:
                self.data.iloc[:, index-1] = self.data.iloc[:, index-1] * 1000
                self.units = 'ppb'
            else:
                self.data.iloc[:, index-1] = self.data.iloc[:, index-1] / 1000
                self.units = 'ppm'
    

    def get_descriptions_OSNACA(self, filename):
        """
        Gets the descriptions at the end of an OSNACA dataset and returns a
        string with new lines for empty cells in between the text.
        """

        # Read in column that only has descriptions at bottom
        first_column = pd.read_excel(filename, usecols=[0], na_values=' ')
        # Remove empty cells
        first_column.fillna(0, inplace=True)

        reached_description = False
        result = ''
        for index, cell in first_column.iterrows():
            if reached_description:
                if cell[0] == 0:
                    result += '\n'
                else:
                    result += cell[0]
                    result += '\n'
            else:
                if cell[0] != 0:
                    reached_description = True
                    result += cell[0] + '\n'
        return result


    def merge_OSNACA(self, cleaner2):
        """
        Takes another OSNACA cleaner and merges the data from it to this one.
        """
        # Append the meta data to this cleaner
        self.descriptions += '\n' + cleaner2.descriptions
        self.title += ' + ' + cleaner2.title

        # Merge the actual data
        self.data = self.data.merge(cleaner2.data)
        

    ### General methods, not as fleshed out


    def remove_columns(self, columns):

        # Check that there are some columns to remove
        if columns is None:
            raise Exception("At least one column must be specified.")

        self.data.drop(columns=columns, inplace=True)
    

    def remove_duplicate_entries(self, subset=None, keep='first', inplace=True):
        """
        Removes duplicate entries from dataframe. Optional arguments are the
        same as designed in the pandas method.

        subset: columns to consider with duplicates
        keep: keep the 'first', 'last' or none (False)
        inplace: do it in place
        """
        return self.data.drop_duplicates(subset=subset, keep=keep, 
                inplace=inplace)


    def remove_entries(self, entries, inplace=True):
        """
        Deletes entries from the dataset, described via a list of their 
        indices.
        """

        # Check that input is a list
        if isinstance(entries, list):

            # Check that each element is an int
            for entry in entries:
                if not isinstance(entry, int):
                    raise TypeError("Entry {} is not an int.".format(entry))

            return self.data.drop(entries, inplace=inplace)   
        else:
            raise TypeError("Index needs to be a list of integers")


    def remove_empty_entries(self, by_row=True, how='any', inplace=True):
        """
        Removes all entries that are considered empty. If by_row is True, 
        each row with at least one empty value will be deleted. False will
        delete in the same way but by column instead.
        """
        if by_row:
            return self.data.dropna(how=how, inplace=inplace)
        else:
            return self.data.dropna(axis='columns', how=how, inplace=inplace)


    def replace_empty_entries(self, values, inplace=True):
        """
        Replaces empty entries within the data. values is a dictionary of
        {column: value} pairs which specifies what value to set each column to.
        Unspecified columns will be ignored.
        """
        if values is None:
            raise Exception("Values is not empty")

        if not isinstance(values, dict):
            raise TypeError("Values must be a dictionary pair")

        self.data.fillna(value=values, inplace=inplace)


    def merge_datasets(self, filenames, extending_rows=True):
        """
        Takes a list of filenames and appends them to the current dataset.
        """
        if filenames is None:
            raise Exception("Must have at least one filename.")
        if not isinstance(filenames, list):
            raise TypeError("Filenames must be a list.")
        
        files = []
        for filename in filenames:
            try:
                this_file = pd.read_csv(filename)
                files.append(this_file)
            except Exception: 
                continue


        if extending_rows:
            self.data = pd.concat([self.data] + files)
        else:
            self.data = pd.concat([self.data] + files, axis='columns')


    def set_data_type(self, columns, data_type):
        """Converts given columns to a specific data type."""
        for column in columns:
            self.data[column] = self.data[column].astype(data_type)


    def write_csv(self, filename=None):
        content = io.BytesIO()
            # # Excel file writer
            #writer = pd.ExcelWriter(filePath+fileName, engine='xlsxwriter')
        # writer = pd.ExcelWriter(output , engine='xlsxwriter')
        # self.data.to_csv(content)

        with gzip.open(content, 'wb') as f:
            f.write(self.data.to_csv().encode())
        #content.seek(0)
        return content
