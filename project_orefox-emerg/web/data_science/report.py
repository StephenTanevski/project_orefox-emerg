from fpdf import FPDF
from fpdf import Template
from numpy.lib.arraysetops import intersect1d
import pandas as pd

from . import analyser
from . import plotter

import os
import io

from django.conf import settings


class PDF(FPDF):
    """
    Subclass of FPDF to define some of the standard sections of the reports.
    """

    def __init__(self, title: str='', title_font: str='Times',
            left_margin: float=10, right_margin: float=10,
            top_margin: float=10):
        """Sets some important metavariables.
        
        Args:
            title: The title to set for the report in the header.
            title_font: The font family for the title.
            left_margin: The margin from the left in mm.
            right_margin: The margin from the top in mm.
            top_margin: The margin from the top in mm.
        """

        super().__init__()

        # Set args as class attributes for the other functions of this class
        self.title = title
        self.title_font = title_font

        # Set margins for this pdf
        self.set_margins(left=left_margin, top=top_margin, 
                right=right_margin)


    def header(self):
        """Defines the header for the pdf."""

        # Logo in top left corner
        self.image( os.path.join(settings.BASE_DIR, 'data_science', 'static', 'data_science', 'resources','images', 'orefoxlogo.png'), y=10, w=35) 

        # Display title
        self.set_font(self.title_font, 'B', size=20)
        self.cell(0, 10, self.title, align='C')

        # Set a line break after the header
        self.ln(20) # TODO Make this customisable?

    
    def footer(self):
        """Defines the footer for the pdf."""

        # 15mm from bottom
        self.set_y(-15)

        self.set_font(self.title_font, 'I')

        # Display a page count
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')



class ReportMaker:
    """Handles process of generating reports."""

    def __init__(self, temp_filepath: str='test/reports/temporary/',
            image_width: float=120, process_obj=None):

        # TODO Remove and create temp folder directory?
        self.temp_filepath = temp_filepath
        self.image_width = image_width
        self.process_obj = process_obj


    def get_x_value_image_centre(self, image_width: float):
        """Finds centre for an image. Useful for centering images.
        
        Takes an image's width and finds the horizontal coordinate for the
        top left by taking the middle of the document and taking half of the
        width from that number.

        Args:
            image_width: The width of the image in inches (matplotlib default).
        """

        # A4 pages are 210mm wide, with 10mm margins on either side
        midpoint = 95 # 190/2

        # Convert image width to mm and divide by two
        distance_from_midpoint = image_width/2

        return 10 + midpoint - distance_from_midpoint


    def table(self, pdf, data):
        """Helper method to display a table.
        
        Args:
            pdf: The pdf to put the table in.
            data: A two dimensional structure of data.
        """
        # TODO Make width and centering optional
        line_height = pdf.font_size * 2.5

        if isinstance(data, pd.DataFrame):
            col_width = (210 - 20) / data.shape[1]
        else:
            col_width = (210 - 20) / len(data)

        if isinstance(data, dict):
            # Column headers
            for column in data:
                pdf.multi_cell(col_width, line_height, 
                        str(column), border=1)

            # Fill in actual data now
            for column in data:
                pdf.multi_cell(col_width, line_height, 
                        str(data[column]), border=1)


    def display_analyser_stats(self, pdf, analyser):
        """Helper function to display analyser.stats in report.
        
        Args:
            pdf: The pdf to display the stats in.
            analyser: The analyser to get the stats from.
        """
        pdf.add_page()

        # Feature count
        feature_string = 'The number of features: {}'.format(
            analyser.stats['num_features'])
        pdf.cell(0, h=10, txt=feature_string, ln=1)

        # Row count
        row_string = 'The number of rows: {}'.format(
            analyser.stats['num_entries']
        )
        pdf.cell(0, h=10, txt=row_string, ln=1)

        # Empty cells
        empty_string = 'The number of empty cells: {}'.format(
            analyser.stats['num_empty']
        )
        pdf.cell(0, h=10, txt=empty_string, ln=1)
        
        # Size of data
        size_string = 'The size of the data: {} bytes'.format(
            analyser.stats['data_size']
        )
        pdf.cell(0, h=10, txt=size_string, ln=1)


    def display_meta_variables(self, pdf, cleaner):

        # Client
        if cleaner.client is not None:
            pdf.cell(0, h=10, txt='Client: ' + cleaner.client, ln=1)
        
        # Date received
        if cleaner.date_received is not None:
            pdf.cell(0, h=10, txt='Date Received: ' + \
                        cleaner.date_received.strftime('%d/%m/%Y'), 
                    ln=1)

        # Date finalised
        if cleaner.date_finalised is not None:
            pdf.cell(0, h=10, txt='Date Finalised: ' + \
                        cleaner.date_finalised.strftime('%d/%m/%Y'),
                    ln=1)

        # Project
        if cleaner.project is not None:
            pdf.cell(0, h=10, txt='Project: ' + cleaner.project, ln=1)

        # Certificate Comments
        if cleaner.certificate_comments is not None:
            pdf.cell(0, h=10, 
                txt='Certificate Comments: ' + cleaner.certificate_comments, 
                ln=1)

        # PO Number
        if cleaner.PO_Number is not None:
            pdf.cell(0, h=10, txt='PO Number: ' + cleaner.PO_Number, ln=1)


    def display_random_forest(self, pdf, plotter):
        """Displays random forest results in report.
        
        Args:
            pdf: The pdf to display in.
            plotter: The plotter that is used to create the figures. This must
              be created with the appropriate analyser, in the space calling
              this helper function.
        """

        # FIGURES
        # TODO Display testing performance as well for rf here.
        pdf.add_page()

        # TODO Add option for temp filepath to be specified
        rf_filename = os.path.join(self.temp_filepath, 'rf.png')

        # TODO Add option to not save file and just return in memory?
        plotter.plot_rf_importances(rf_filename, big_to_small=False, 
                figsize=(14,14))

        pdf.image(rf_filename, 
                x=self.get_x_value_image_centre(self.image_width),
                w=self.image_width)

        plotter.plot_model_predictions(
                plotter.analyser.rf_test_predictions, 
                plotter.analyser.rf_test_y, 
                os.path.join(self.temp_filepath, 'rf_performance.png')
            )

        pdf.image(os.path.join(self.temp_filepath, 'rf_performance.png'),
                x=self.get_x_value_image_centre(self.image_width),
                w=self.image_width)

        # TEXT TODO Make this an option
        pdf.add_page()

        with open('src/resources/infos/rf.txt', 'r') as f:
            rf_info = f.read().replace('\n', '')

        pdf.multi_cell(0, h=5, txt=rf_info)


    def display_pca(self, pdf, plotter):
        """Display PCA results in report.
        
        Args:
            pdf: The pdf to display in.
            plotter: The plotter that is used to create figures. This must
              be created with the appropriate analyser, in the space calling
              this helper function.
        """

        # FIGURES
        pdf.add_page()

        pca_filenames = [self.temp_filepath + 'pca.png',
                self.temp_filepath + 'pca2.png']

        plotter.plot_pca_feature_bar(pca_filenames[0], figsize=(14,14))
        plotter.plot_pca_cumulative_importance(pca_filenames[1], 
                figsize=(14,14))

        pdf.image(pca_filenames[0],
                x=self.get_x_value_image_centre(self.image_width),
                w=self.image_width)

        pdf.image(pca_filenames[1],
                x=self.get_x_value_image_centre(self.image_width),
                w=self.image_width)

        # TEXT

        pdf.add_page()

        with open('src/resources/infos/pca.txt', 'r') as f:
            pca_info = f.read().replace('\n', '')

        pdf.multi_cell(0, h=5, txt=pca_info)

        with open('src/resources/infos/pca2.txt', 'r') as f:
            pca_cumulative_info = f.read().replace('\n', '')

        pdf.multi_cell(0, h=5, txt=pca_cumulative_info)


    def display_hca(self, pdf, plotter): 
        """Display HCA results in report.
        
        Args:
            pdf: The pdf to display in.
            plotter: The plotter that is used to create figures. This must
              be created with the appropriate analyser, in the space calling
              this helper function.
        """
        
        # FIGURES

        pdf.add_page()

        # TODO Add option for temp filepath to be specified
        hca_filename = self.temp_filepath + 'hca.png'

        # TODO Add option to not save file and just return in memory?
        # TODO Labels need to be set here
        plotter.plot_dendrogram(hca_filename, 
                labels=plotter.analyser.datacleaner.data.columns,
                figsize=(14,14), label_font_size=8, rotation=90)

        pdf.image(hca_filename, 
                x=self.get_x_value_image_centre(self.image_width),
                w=self.image_width)

        # TEXT
        pdf.add_page()

        with open('src/resources/infos/hca.txt', 'r') as f:
            hca_info = f.read().replace('\n', '')

        pdf.multi_cell(0, h=5, txt=hca_info)


    def display_cc(self, pdf, plotter):
        """Display HCA results in report.
        
        Args:
            pdf: The pdf to display in.
            plotter: The plotter that is used to create figures. This must
              be created with the appropriate analyser, in the space calling
              this helper function.
        """
        pdf.add_page()

        with open(os.path.join(settings.BASE_DIR, 'data_science', 'static', 'data_science', 'resources','infos', 'cc.txt'), 'r') as f:
            cc_info = f.read().replace('\n', '')

        pdf.multi_cell(0, h=5, txt=cc_info)

        # cc_filename = [ os.path.join(settings.BASE_DIR, 'result', 'cc1.png'), 
        #                 os.path.join(settings.BASE_DIR, 'result', 'cc1.png'), 
        #                 os.path.join(settings.BASE_DIR, 'result', 'cc1.png'),  ]

        # plotter.plot_correlations_heatmap(filenames=cc_filename)

        process_files = self.process_obj.process_files.filter(stage_name = 'plot', stage_action_name = 'cc_heatmap') # my edit

        cc_filename = []
        for pf in process_files:
            cc_filename.append(pf.file.path)

        cc_image_width = self.image_width - 10

        pdf.image(cc_filename[0], 
                x=self.get_x_value_image_centre(cc_image_width),
                w=cc_image_width)

        pdf.image(cc_filename[1], 
                x=self.get_x_value_image_centre(cc_image_width),
                w=cc_image_width)

        pdf.add_page()

        pdf.image(cc_filename[2], 
                x=self.get_x_value_image_centre(cc_image_width),
                w=cc_image_width)


    def display_adaboost(self, pdf, plotter):
        """Display AdaBoost results in report.
        
        Args:
            pdf: The pdf to display in.
            plotter: The plotter that is used to create figures. This must
              be created with the appropriate analyser, in the space calling
              this helper function.
        """

        # FIGURES
        pdf.add_page()

        plotter.plot_adaboost_importances(self.temp_filepath + 'ada.png',
                big_to_small=False, figsize=(14,14))

        pdf.image(self.temp_filepath + 'ada.png', 
                x=self.get_x_value_image_centre(self.image_width),
                w=self.image_width)

        plotter.plot_model_predictions(
                plotter.analyser.ada_test_predictions, 
                plotter.analyser.ada_test_y, 
                self.temp_filepath + 'ada_performance.png'
            )

        pdf.image(self.temp_filepath + 'ada_performance.png',
                x=self.get_x_value_image_centre(self.image_width),
                w=self.image_width)

        # TEXT 
        pdf.add_page()

        with open('src/resources/infos/adaboost.txt', 'r') as f:
            ada_info = f.read().replace('\n', '')

        pdf.multi_cell(0, h=5, txt=ada_info)


    def display_xgboost(self, pdf, plotter):
        """Display XGBoost results in report.
        
        Args:
            pdf: The pdf to display in.
            plotter: The plotter that is used to create figures. This must
              be created with the appropriate analyser, in the space calling
              this helper function.
        """

        # FIGURES
        pdf.add_page()

        plotter.plot_xgboost_importances(self.temp_filepath + 'xg.png',
                big_to_small=False, figsize=(14,14))

        pdf.image(self.temp_filepath + 'xg.png', 
                x=self.get_x_value_image_centre(self.image_width),
                w=self.image_width)

        plotter.plot_model_predictions(
                plotter.analyser.xgboost_test_predictions, 
                plotter.analyser.xgboost_test_y, 
                self.temp_filepath + 'xg_performance.png'
            )

        pdf.image(self.temp_filepath + 'xg_performance.png',
                x=self.get_x_value_image_centre(self.image_width),
                w=self.image_width)

        # TEXT 
        pdf.add_page()

        with open('src/resources/infos/xgboost.txt', 'r') as f:
            ada_info = f.read().replace('\n', '')

        pdf.multi_cell(0, h=5, txt=ada_info)


    def display_rf_ada_xg_compare(self, pdf, plot):
        """Display comparison of RF and AdaBoost in report.
        
        Args:
            pdf: The pdf to display in.
            plotter: The plotter that is used to create figures. This must
              be created with the appropriate analyser, in the space calling
              this helper function.
        """
        intersect = plot.analyser.compare_ada_rf_xg()

        pdf.add_page()

        if len(intersect) == 0:
            pdf.multi_cell(0, h=5, txt='There are no features in common.')
            # TODO Add some explanation here

        else:
            pdf.multi_cell(0, h=5, txt='The features in common are:')

            for feature in intersect:
                pdf.multi_cell(0, h=5, txt='    ' + feature)


    def make_analysis_report(self, analyser, filepath: str):
        """Makes a report based on analysis done.
        
        Args:
            analyser: The analyser to generate a report on.
            filepath: The output filepath for the pdf report.
        """

        pdf = PDF(title='Analysis Report')
        pdf.alias_nb_pages()
        pdf.set_font('Times', '', 12)

        plot = plotter.Plotter(analyser)

        if analyser.stats is not None:
            self.display_analyser_stats(pdf, analyser)

        if analyser.rf is not None:
            self.display_random_forest(pdf, plot)

        if analyser.adaboost is not None:
            self.display_adaboost(pdf, plot)

        if analyser.xgboost is not None:
            self.display_xgboost(pdf, plot)

        if analyser.rf is not None and analyser.adaboost is not None and \
                analyser.xgboost is not None:
            self.display_rf_ada_xg_compare(pdf, plot)

        if analyser.pca is not None:
            self.display_pca(pdf, plot)

        if analyser.hca is not None:
            self.display_hca(pdf, plot)

        if analyser.correlations is not None:
            self.display_cc(pdf, plot)

        # content = io.BytesIO()
        content = pdf.output(dest='S').encode('latin-1')
        expected_filename = filepath[:-(len(filepath.split('.')[-1])+1)]
        filename = filepath
        data = {
            'content': content,
            'expected_filename': expected_filename,
            'filename': filename
        }
        # TODO Clear temp folder?
        return data


    def make_data_report(self, datacleaner, filepath: str):
        """Makes a report based on the data in the given datacleaner.
        
        Args:
            datacleaner: The datacleaner that contains the data to report.
            filepath: The place to save the pdf to.
        """

        # Option to put line at certain threshold for missing values? TODO

        # Initialise pdf, analyser and plotter
        pdf = PDF(title='Data Report')
        pdf.alias_nb_pages()
        pdf.set_font('Times', '', 12)

        a = analyser.Analyser(datacleaner)
        plot = plotter.Plotter(a)

        # Get stats as of now
        a.get_dataset_stats()
        
        # Display stats of dataset as well as metavariables if there
        self.display_analyser_stats(pdf, a)
        self.display_meta_variables(pdf, datacleaner)

        # Print number of missing cells per column, line by line
        pdf.add_page()

        counter = 0
        for column in datacleaner.data.columns:
            string = 'Number of missing values in {}: {}'.format(column,
                    str(a.stats['empty_cells_per_column'][counter]))
            pdf.cell(0, h=10, txt=string, ln=1)
            counter += 1

        #self.table(pdf, a.stats['empty_cells_per_column'])

        filename = self.temp_filepath + 'empty.png'

        plot.visualise_empty_cells(filename, figsize=(14,14), rotation=90)

        # Visualisations, bar and scatter, for empty cells per column
        pdf.add_page()

        pdf.image(filename, 
                x=self.get_x_value_image_centre(self.image_width),
                w=self.image_width)

        filename = self.temp_filepath + 'empty_bar.png'

        plot.visualise_empty_cells_bar(filename, figsize=(14,14), rotation=90)

        pdf.image(filename, 
                x=self.get_x_value_image_centre(self.image_width),
                w=self.image_width)

        # Data types
        pdf.add_page()

        for column in datacleaner.data.columns:
            string = '{} is of type {}.'.format(column,
                    datacleaner.data[column].dtype)
            pdf.cell(0, h=10, txt=string, ln=1)

        pdf.output(filepath, 'F')


    def make_cleaner_report(self, datacleaner, filename: str):
        """Makes a report based on the cleaning done.

        Args:
            datacleaner: The DataCleaner to generate a report on.
            filename: The output filename for the pdf.
        """
        # Initialise pdf, analyser and plotter
        pdf = PDF(title='Cleaning Report')
        pdf.alias_nb_pages()
        pdf.set_font('Times', '', 12)

        with open('logs/logger_' + datacleaner.username + '.log', 'r') as f:
            log = f.readlines()

        for line in log:
            # If columns have been removed
            if 'REMOVE_COLUMNS:' in line:
                string = line.split('REMOVE_COLUMNS:')[-1]
                
                pdf.add_page()

                columns = string.split(',')
                if len(columns) == 0:
                    pdf.multi_cell(0, h=5, txt='No Columns Removed.')
                else:
                    pdf.multi_cell(0, h=5, txt='Columns Removed:')
                    for column in columns:
                        pdf.multi_cell(0, h=5, txt='    ' + column)
            
            # If imputing values was done
            elif 'IMPUTE_VALUES' in line:
                string = line.split('IMPUTE_VALUES:')[-1]

                # Start after the MODE word in log
                counter = 4
            
                # Construct mode string
                mode = ''
                while True:
                    if string[counter] == ':':
                        break
                    mode += string[counter]
                    counter += 1

                # Get only columns now
                string = line.split('IMPUTE_VALUES:MODE' + mode + ':')[-1]
                columns = string.split(',')

                # Print columns
                if len(columns) == 0:
                    pdf.multi_cell(0, h=5, txt='No columns imputed.')
                else:
                    pdf.multi_cell(0, h=5, txt='Columns Imputed ' + \
                            'using {}:'.format(mode))

                    for column in columns:
                        pdf.multi_cell(0, h=5, txt='    ' + column)

            # If handling the inequality cells
            elif 'HANDLE_INEQUALITIES' in line:
                string = line.split('HANDLE_INEQUALITIES:')[-1]

                columns = string.split(',')

                if len(columns) == 0:
                    pdf.multi_cell(0, h=5, txt='No columns were searched ' + \
                            'for < or > symbols.')
                else:
                    pdf.multi_cell(0, h=5, txt='Columns checked for < or >' + \
                            ' cells')

                    for column in columns:
                        pdf.multi_cell(0, h=5, txt='    ' + column)

            # If converting units
            elif 'UNITS' in line:
                string = line.split('UNITS:')[-1]

                pdf.multi_cell(0, h=5, txt='Columns were converted to {}'.format(
                        string))

            # Removal of duplicate units
            elif 'DUPLICATES' in line:
                string = line.split('DUPLICATES:')[-1]

                pdf.multi_cell(0, h=5, txt=string)

            # Removing entries
            elif 'REMOVE_ENTRIES' in line:
                string = line.split('REMOVE_ENTRIES:')[-1]

                pdf.multi_cell(0, h=5, txt=string)

            # Writing to file
            elif 'WRITE_CSV' in line:
                pdf.multi_cell(0, h=5, txt='Wrote to file.')


        # Save pdf
        pdf.output(filename, 'F')