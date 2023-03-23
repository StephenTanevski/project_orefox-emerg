"""Holds a class for an interface that loads and manipulates data."""

from datetime import date
import math
import os

import miceforest as mf
import numpy as np
import pandas as pd
from sklearn import preprocessing

from django.conf import settings
import os

from . import logger

class DataCleaner:
    """Holds methods for importing and manipulating data.
    
    Attributes:
        data: pd.DataFrame of the data loaded from filename
        logger: Instance of logger class, used to log events of this class
        title: String of title, uses either given meta data or filename
        client: String of client
        num_samples: The number of samples taken as an int
        date_received: The date the samples were received, as a DateTime
        date_finalised: The date the data was finalised, as a DateTime
        project: The name of the project
        certificate_comments: Certificate comments in data
        PO_number: The PO number of the project
        units: A string representation of the units used in the data
    """

    def __init__(self, filename: str, testing: bool=False, lab: str=None,
            unit: str='ppb', index_col: int=None, null_replace: bool=None,
            sort_columns_OSNACA: bool=True, fillna_Petrosea: bool=True,
            username: str='Default', fillna_mode: str='zero', 
            low_memory_load: bool=False, is_excel: bool=False, 
            is_xls: bool=True, sheet_name: int=0):
        """Handles loading of data from filename.

        
        This currently supports data from ALS, OSNACA and Petrosea. If not
        specifying a particular lab, the data must have the first row as 
        column headers, and have a normal structure to the data table.

        Args:
            filename: String of the path to the data to be loaded
            testing: If True, only load first 5000 lines, False, load all
            lab: String representation to indicate lab presets
            unit: Unit to use for uniform units in {ppm, ppb, pc}.
            index_col: The row number that contains the column headers
            null_replace: If True, replaces the empty values
            sort_columns_OSNACA: If True, this will sort the columns of the
              element data in OSNACA lab data. Setting to False is important 
              for some tests.
            fillna_Petrosea: If True, replaces null cells in Petrosea data
            username: The username of the user, for logging purposes.
            fillna_mode: The method in how values should be imputed.
            low_memory_load: Same as in pd.read_csv, whether or not to load the
              data in as chunks (set to False), or just all as one (set to 
              True).
            is_excel: Input file is not csv, but rather a .xls file or a
              similar type of 'excel' file.
            is_xls: More specifically, if the file is a .xls file. If this is
              the case, the C engine can be used.
            sheet_name: For read_excel, the sheet of the spreadsheet to load
              in.

        Raises:
            FileNotFoundError: An error occurred by loading a file that cannot
              be found
        """

        # Define a logger per user
        self.logger = logger.Logger('logger_{}'.format(username))

        self.username = username

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
                        index_col=index_col, low_memory=low_memory_load)
            else:
                if is_excel:
                    # Note no low_memory option is supported with the Python
                    # engine
                    if is_xls:
                        self.data = pd.read_excel(filename, index_col=index_col,
                                sheet_name=sheet_name, 
                                low_memory=low_memory_load)
                    else:
                        self.data = pd.read_excel(filename, index_col=index_col,
                                sheet_name=sheet_name, engine='openpyxl')
                else:
                    self.data = pd.read_csv(filename, index_col=index_col,
                            low_memory=low_memory_load)
            
            # Set the number of samples based on shape of data
            self.num_samples = self.data.shape[0]

            # Send alert to us just in case it needs supervision
            # TODO Make it send an email
            self.logger.logger.info('POTENTIAL_LEGACY_DATA')
                
        # Data from ALS
        elif lab == 'ALS':
            # Get the meta-variables listed at the beginning of the file
            meta_data = pd.read_csv(filename, nrows=7, usecols=[0], 
                    header=None)
            self.set_meta_variables(meta_data)
            
            # Read in the data
            self.data = pd.read_csv(
                filename,
                index_col=0,
                skiprows=7,
                header=1,
                low_memory=low_memory_load
            )

            # Add method to headers to reduce the header to one row
            self.update_headers_ALS(filename)

            # Deal with cells that have < or > in them
            self.handle_inequalities()

            # Convert to same units
            self.get_units(0)
            self.convert_uniform_units(unit=unit, update_col_suffix=False)

            self.set_data_type(self.data.columns, np.float32)

        # Data from OSNACA
        elif lab == 'OSNACA':
            self.data = pd.read_excel(
                filename,
                index_col=1,
                na_values=[' ', 'MISSING', 'IS', 'TF', 'DMGED', 
                        'empty', 'CAVITY'],
                load_memory=low_memory_load
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

            # Make this a conditional just to help some of the testing
            if sort_columns_OSNACA:
                self.make_element_columns_alphabetical_OSNACA()

            # Set the number of samples based on shape of data
            self.num_samples = self.data.shape[0]

        elif lab == 'Petrosea':

            # Set two of the final columns to strings
            DTYPES = {78: str, 86: str}

            # Load the original file
            self.data = pd.read_csv(
                filename, 
                index_col=1, # Use the hole id as the key
                na_values=[' ', 'SNR', 'NS'], # Disregard these symbols
                dtype=DTYPES,
                nrows=5000 if testing else None,
                low_memory=low_memory_load
            )

            # Set the number of samples based on shape of data
            self.num_samples = self.data.shape[0]

        elif lab == 'Petrosea-Mixed':

            # Set two of the final columns to strings
            DTYPES = {78: str, 86: str}

            # The column indexes that contain < symbols
            inequality_columns = [68, 69, 70, 71, 72, 73, 75, 76]

            self.data = pd.read_csv(
                filename, 
                index_col=1, # Use the hole id as the key
                na_values=[' ', 'SNR', 'NS'], # Disregard these symbols
                dtype=DTYPES,
                nrows=5000 if testing else None,
                low_memory=low_memory_load
            )

            # Set the number of samples based on shape of data
            self.num_samples = self.data.shape[0]

            column_names = [self.data.columns[i] for i in range(6, 50)]
            if fillna_Petrosea:
                self.replace_empty_entries(mode=fillna_mode, 
                        subset_X=column_names)

            self.handle_inequalities(use_columns=inequality_columns)

            self.get_units(0, suffix=True, drop_units_row=False)
            self.convert_uniform_units(unit=unit, subset_X=column_names)

        self.logger.logger.info('Loaded {} data.'.format(self.lab))


    def __repr__(self) -> str:
        """String representation of this class."""
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


    def set_meta_variables(self, meta_data: list) -> None:
        """Method to handle initialisation of meta-variables. 
        
        Currently onlyuseful for ALS because these are specified in the 
        document.

        Args:
            meta_data: List of the meta data read from an ALS datasheet.
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


    def get_units(self, units_row_index: int, suffix: bool=False,
            sep: str='_', drop_units_row: bool=True) -> None:
        """Extracts units information from data.
        
        Args:
            units_row_index: The row index that the units info is in.
            suffix: True if the units are at the end another string in that
              row, for example, Ag_ppm instead of ppm.
            sep: The separator that is used for the string to separate the
              the other text from the unit.
            drop_units_row: True will drop the row from the data frame.
        """

        result = {}

        for column in self.data.columns:
            # Case where units are appended at end of element name
            if suffix:
                unit = column.split(sep)[-1]
            else:
                unit = self.data[column].iloc[units_row_index]
                
            result[column] = unit

        self.units = result

        if drop_units_row:
            self.data.drop(self.data.index[units_row_index], inplace=True)


    def update_headers_ALS(self, filename: str) -> None:
        """Merges the header info from an ALS dataset into one row.

        Helper function that takes data from ALS and appends the method of
        measurement to the end of the element, so that the header of each
        column is one cell.

        Args:
            filename: String of the path of the original data.
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


    def update_headers_OSNACA(self, filename: str) -> None:
        """Merges the header info from an OSNACA dataset into one row.

        Helper function that takes data from OSNACA and appends the method of
        measurement to the end of the element, so that the header of each
        column is one cell.

        Args:
            filename: String of the path of the original data.
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


    def handle_inequalities(self, use_columns: list=None, 
            set_inequality_tolerance: bool=False) -> None:
        """Replaces inequality cells.

        Replaces the cells that involve inequalities rather than specific
        measurements.

        Args:
            use_columns: List of columns to look at, as column indexes
            set_inequality_tolerance: If True, sets the cells to the
              tolerance set, False sets to zero.
        """
        # Get the columns to use
        if use_columns is None:
            columns = range(self.data.shape[1])
        else:
            # Note that this assumes that the columns are all connected
            columns = use_columns

        # Most likely not the most efficient way to find all < cells
        for column in columns:
            for row in range(self.data.shape[0]):

                # Check that this cell is a string
                if not isinstance(self.data.iat[row, column], str):
                    continue

                # Check which inequality it is
                if self.data.iat[row, column][0] == '<':
                    if set_inequality_tolerance:
                        # Just remove the '< ' from the tolerance
                        if self.lab == 'ALS':
                            self.data.iat[row, column] = \
                                    self.data.iat[row, column][1:]
                        elif self.lab == 'Petrosea' or \
                                    self.lab == 'Petrosea-Mixed':
                            self.data.iat[row, column] = \
                                    self.data.iat[row, column][2:]
                    else:
                        self.data.iat[row, column] = 0
                elif self.data.iat[row, column][0] == '>':
                    # Set to tolerance
                        if self.lab == 'ALS':
                            self.data.iat[row, column] = \
                                    self.data.iat[row, column][1:]
                        elif self.lab == 'Petrosea' or \
                                self.lab == 'Petrosea-Mixed':
                            self.data.iat[row, column] = \
                                    self.data.iat[row, column][2:]

        # Log action to file
        string = 'HANDLE_INEQUALITIES:'
        first_loop = True
        counter = 0
        for column in self.data.columns:

            # Skip if not in used columns
            if counter not in columns:
                counter += 1
                continue
            
            if first_loop:
                string += str(column)
                first_loop = False
            else:
                string += ',' + str(column)

            counter += 1

        self.logger.logger.info(string)


    def convert_uniform_units(self, unit: str='ppm', 
            subset_X: list=None, update_col_suffix: bool=True, 
            append_suffix: bool=False) -> None:
        """This converts all ppm/ppb/pc values to one of these.
        
        Assumes that all values are either ppm, ppb or something that does 
        not need to be converted because it is already uniform.

        Args:
            unit: The unit to convert to. Must be one of {ppm, ppb, pc}.
            subset_X: List of columns to use, if None, use all columns.
            update_col_suffix: True will update the column header at the end
              to represent the unit that column is now in.
            append_suffix: If True, adds the unit to the end of the column,
              False will replace the last split (_ as a delimiter).
        """

        # Cast to floats
        if subset_X is not None:
            self.data[subset_X] = self.data[subset_X].astype(np.float32)
        else:
            self.data = self.data.astype(np.float32)

        for column in self.units:
            # Ignore columns that are not in subset_X
            if subset_X is not None:
                if not column in subset_X:
                    continue

            # Already in the correct unit
            if self.units[column] == unit:
                continue

            if unit == 'ppb':
                if self.units[column] == 'ppm':
                    # Convert from ppm to ppb
                    self.data[column] = self.data[column] * 1000
                elif self.units[column] == 'pc':
                    # Convert from pc to ppb
                    self.data[column] = self.data[column] * 10000000 

            elif unit == 'ppm':
                if self.units[column] == 'ppb':
                    # Convert from ppb to ppm
                    self.data[column] = self.data[column] / 1000
                elif self.units[column] == 'pc':
                    # Convert from pc to ppm 
                    self.data[column] = self.data[column] * 10000

            elif unit == 'pc':
                if self.units[column] == 'ppm':
                    # Convert from ppm to pc
                    self.data[column] = self.data[column] / 10000
                elif self.units[column] == 'ppb':
                    self.data[column] = self.data[column] / 10000000
            
            if unit == 'kg' or unit == 'Kg':
                continue
            
            self.units[column] = unit
            if update_col_suffix:
                # Get updated string
                if append_suffix:
                    self.data.rename(columns={column: column + unit}, 
                            inplace=True)
                else:
                    # Get first parts of the string
                    result = ''
                    split = column.split('_')
                    if len(split) == 1:
                        result += split[0] + '_'
                    else:
                        for string in split[:-1]:
                            result += string + '_'

                    # Add new unit to string, and update column name
                    result += unit
                    self.data.rename(columns={column: result},
                            inplace=True)

        self.logger.logger.info('UNITS: {}'.format(unit))
    

    def get_descriptions_OSNACA(self, filename: str) -> str:
        """Gets the descriptions at the end of an OSNACA dataset 
        
        Args:
            filename: The file path of the original dataset

        Returns:
            The concatenated string of descriptions
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


    def make_element_columns_alphabetical_OSNACA(self) -> None:
        """Sorts element columns to be in alphabetical order for OSNACA data.

        This is only for the elements of the data.
        """
        new_columns = np.sort(self.data.columns[:-1])
        self.data = self.data.reindex(columns=new_columns)


    def merge_OSNACA(self, cleaner2) -> None:
        """Takes another OSNACA cleaner and merges the data from it.

        Args:
            cleaner2: The other data, as another datacleaner, to merge.
        """
        # Append the meta data to this cleaner
        self.descriptions += '\n' + cleaner2.descriptions
        self.title += ' + ' + cleaner2.title

        # Merge the actual data
        self.data = self.data.merge(cleaner2.data)


    def split_Petrosea(self, filenames: list, 
            min_elements_mixed: int=3) -> None:
        """Takes original Petrosea data and splits it into three files.
        
        Args:
            filenames: A list of three strings, corresponding to (in order)
              au only, mixed and au plus some.
            min_elements_mixed: The minimum number of elements that must be
              non zero to be placed into the mixed set rather than the au plus
              some set
        """

        # The index of columns that hold "other" elements
        columns_other_elements = [x for x in range(7, 67)]


        # Dataframes that the original will be split into
        au_only = pd.DataFrame(columns=self.data.columns)
        mixed_elements = pd.DataFrame(columns=self.data.columns)
        au_plus_some = pd.DataFrame(columns=self.data.columns)

        row_counter = 0
        for _, row in self.data.iterrows():
            element_counter = 0
            # See how many elements for this row are positive
            for index in columns_other_elements:
                if not math.isnan(row[index]):
                    element_counter += 1
                    
            # Check progress as it's running
            if row_counter % 10000 == 0:
                self.logger.logger.info('Splitter completed {} rows.'.format(
                        row_counter))

            # Sort accordingly into the different spreadsheets
            if element_counter >= min_elements_mixed:
                mixed_elements = mixed_elements.append(row.to_dict(), 
                        ignore_index=True)
            elif element_counter == 0:
                au_only = au_only.append(row.to_dict(), 
                        ignore_index=True)
            else:
                au_plus_some = au_plus_some.append(row.to_dict(), 
                        ignore_index=True)

            row_counter += 1

        # Save the current file
        au_only.to_csv(filenames[0])
        mixed_elements.to_csv(filenames[1])
        au_plus_some.to_csv(filenames[2])
        

    ### General methods, not as fleshed out


    def remove_columns(self, columns: list) -> list:
        """Removes the given columns (not None) from the data.
        
        Returns:
            List of columns to be dropped from the table. Logs columns too.
        """
        # TODO Need to instead record the columns actually dropped
        # Can be done by looking at difference between input arg and data.cols

        # Check that there are some columns to remove
        if columns is None:
            raise Exception("At least one column must be specified.")

        self.data.drop(columns=columns, inplace=True)

        # Log the columns to file as well
        string = 'REMOVE_COLUMNS:'
        first_loop = True
        for column in columns:
            if first_loop:
                string += str(column)
                first_loop = False
            else:
                string += ',' + str(column)

        self.logger.logger.info(string)

        return columns
    

    def remove_duplicate_entries(self, subset: list=None, 
            keep: str='first', inplace: bool=True) -> pd.DataFrame:
        """Removes duplicate entries from dataframe, same as pandas."""

        self.logger.logger.info("DUPLICATES:Duplicate entries were removed.")

        return self.data.drop_duplicates(subset=subset, keep=keep, 
                inplace=inplace)


    def remove_entries(self, entries: list, 
            inplace: bool=True) -> pd.DataFrame:
        """Deletes entries from the dataset.
        
        Args:
            entries: List of indexes to remove.
            inplace: Whether or not to perform this in place.

        Returns:
            Dataframe of updated data.

        Raises:
            TypeError: Input entries must be of correct type.
        """

        # Log the attempt to file
        self.logger.logger.info('REMOVE_ENTRIES:Attempted to ' + \
                    'delete {} entries'.format(len(entries)))

        # Check that input is a list
        if isinstance(entries, list):
            return self.data.drop(entries, inplace=inplace)   
        else:
            raise TypeError("Index needs to be a list of integers")


    def remove_empty_entries(self, by_row: bool=True, how: str='any', 
            inplace: bool=True) -> pd.DataFrame:
        """Removes all entries that are considered empty. 
        
        Args:
            by_row: If True, do it by rows, False, by columns
            how: String, options between any and all
            inplace: Whether or not to do this in place.

        Returns:
            Updated data frame.
        """
        if by_row:
            return self.data.dropna(how=how, inplace=inplace)
        else:
            return self.data.dropna(axis='columns', how=how, inplace=inplace)


    def replace_empty_entries(self, mode: str='zero', values: dict=None,
            subset_X: list=None, inplace: bool=True, 
            aca_source: str='CRC', mice_save_filepath: str=None) -> None:
        """Replaces empty entries within the data. 

        Unspecified columns will be ignored.
        
        Args:
            values: {column: value} pairs which specifies what value to set 
              each column to. Types are {string: object}
            mode: 
              - 'zero' will set all empty values to 0.
              - 'aca' will set all element values to average crustal abundance.
                  Note that this uses the aca.csv file for values.
              - 'median' will set empty vales to the median of the column.
              - 'mean' will set empty values to the mean of the column.
              - 'mice' will use multiple imputation by chained equations
              - None will use values parameter.
            subset_X: None uses all columns, else use these columns
            inplace: Whether or not to do this in place.
            mice_save_filepath: The filepath for all of the MICE outputs.
        """
        # Get list of columns to use
        columns = subset_X if subset_X is not None else self.data.columns

        # Begin string to log after success

        # Beginning of string
        string = 'IMPUTE_VALUES:MODE{}:'.format(mode)
        
        # Columns that are attempted to impute # TODO Make this a helper function in logger
        first_loop = True
        for column in columns:
            if first_loop:
                string += str(column)
                first_loop = False
            else:
                string += ',' + str(column)

        self.logger.logger.info(string)

        # Just fill with specified values
        if mode is None:
            self.data.fillna(value=values, inplace=inplace)
            
        # Fill empty cells with zero
        elif mode == 'zero':
            self.data.fillna(0, inplace=inplace)

        # Fill empty cells with the median of the column
        elif mode == 'median':
            for column in columns:
                self.data[column].fillna(self.data[column].median(), 
                        inplace=inplace)

        # Fill empty cells with the mean
        elif mode == 'mean':
            for column in columns:
                self.data[column].fillna(self.data[column].mean(), 
                        inplace=inplace)
        
        # Fill empty cells with the appropriate average crustal abundance
        elif mode == 'aca':
            # Load in ACA data, indexed by element symbol
            aca = pd.read_csv( os.path.join(settings.BASE_DIR, 'resources', 'aca.csv'), index_col=3)

            # Remove unit appended at end
            elements = []
            for column in columns:
                # Assumes in the form '{symbol}_{unit}'
                elements.append(column.split('_')[0])

            counter = 0
            for column in columns:
                # Skip weight column
                if elements[counter] == 'Wt':
                    continue

                if column[-2:] == 'pc':
                    # Convert the aca to pc to make consistent with column
                    value = aca[aca_source][elements[counter]] / 10000
                else:
                    value = aca[aca_source][elements[counter]]

                self.data[column].fillna(value, inplace=inplace)
                counter += 1

        # Fill empty cells using MICE
        elif mode == 'mice':

            datasets = 5
            kernel = mf.MultipleImputedKernel(
                data=self.data[columns],
                datasets=datasets,
                save_all_iterations=False,
                random_state=10
            )

            iterations = 5
            kernel.mice(iterations, verbose=False)

            if mice_save_filepath is None:
                mice_save_filepath = 'test/testdata/mice_'

            for i in range(datasets):
                kernel.complete_data(i).to_csv(mice_save_filepath + str(i) 
                        + '.csv')


    def merge_datasets(self, filenames: list, 
            extending_rows: bool=True) -> None:
        """Takes a list of filenames and appends them to the current dataset.

        Args:
            filenames: A list of strings that go to each file
            extending_rows: If True, adds rows wise, False, columns wise
        """

        # Check input is as desired
        assert filenames is not None
        assert isinstance(filenames, list)
        
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


    def set_data_type(self, columns: list, data_type: type) -> None:
        """Converts given columns to a specific data type."""
        for column in columns:
            self.data[column] = self.data[column].astype(data_type)


    def write_csv(self, filename: str) -> None:
        """Writes current data to filename (must end with .csv)."""
        self.data.to_csv(filename)

        # Log this to file
        self.logger.logger.info('WRITE_CSV:{}'.format(filename))
