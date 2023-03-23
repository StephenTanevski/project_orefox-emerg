from .analyser import Analyser
from .cleaner import DataCleaner
from .plotter import Plotter
from .report import ReportMaker
from .input_codes import Codes

import os
from django.core.files.base import ContentFile
from django.conf import settings
from appboard.models import ProcessFile


def processor(stage: int, action: int=None, args: list=None, process_obj=None):
    """A function that takes input and calls the necessary function.

    Args:
        stage: The stage of the workflow the user is at. Refer to
          input_codes.WORKFLOW_STAGE for details.
        action: The numerical id of the action to take. These are not unique,
          but with reference to the stage will be unique.
        args: A dictionary of params for the stage/action pair. # TODO Make a checker function for the dictionary
          If a DataCleaner/Analyser/Plotter/ReportMaker is used, it must be the
          first arg in the list.
        cleaner: The DataCleaner to use for cleaning stage.

    Returns:
        DataCleaner: In the case of loading and cleaning stages.
    """
    REV_WORKFLOW_STAGE = Codes.get_reversed_dictionary(Codes.WORKFLOW_STAGE)

    # Loading data in initially
    if stage == Codes.WORKFLOW_STAGE['load_data']:

        # Translate the args so DataCleaner can be created
        input_args = Codes.get_cleaner_constructor_dictionary(args)

        cleaner = DataCleaner(
            filename=input_args['filename'], 
            lab=input_args['lab'],
            unit=input_args['unit'],
            index_col=input_args['index_col'],
            username=input_args['username'],
            is_excel=input_args['is_excel'],
            is_xls=input_args['is_xls'],
            sheet_name=input_args['sheet_name']
        )
        # Keep adding params here

        return cleaner

    # Cleaning
    elif stage == Codes.WORKFLOW_STAGE['clean_data']:
        
        # Handle inequalities
        if action == Codes.CLEANER_ACTIONS['handle_inequalities']:
            args[0].handle_inequalities(
                use_columns=args[1],
                set_inequality_tolerance=args[2]
            )
        
        # Convert units
        elif action == Codes.CLEANER_ACTIONS['convert_uniform_units']:
            args[0].convert_uniform_units(
                unit=args[1], 
                subset_X=args[2], 
                update_col_suffix=args[3], 
                append_suffix=args[4]
            )

        # Remove columns
        elif action == Codes.CLEANER_ACTIONS['remove_columns']:
            args[0].remove_columns(
                columns=args[1]
            )

        # Remove duplicate entries
        elif action == Codes.CLEANER_ACTIONS['remove_duplicate_entries']:
            args[0].remove_duplicate_entries(
                subset=args[1], 
                keep=args[2], 
                inplace=args[3]
            )

        # Remove entries
        elif action == Codes.CLEANER_ACTIONS['remove_entries']:
            args[0].remove_entries(
                entries=args[1], 
                inplace=args[2]
            )

        # Remove empty entries
        elif action == Codes.CLEANER_ACTIONS['remove_empty_entries']:
            args[0].remove_empty_entries(
                by_row=args[1], 
                how=args[2], 
                inplace=args[3]
            )

        # Imputing values
        elif action == Codes.CLEANER_ACTIONS['impute']:
            args[0].replace_empty_entries(
                mode=Codes.CLEANER_IMPUTING_ACTIONS(args[1]), 
                values=args[2],
                subset_X=args[3], 
                inplace=args[4], 
                aca_source=args[5], 
                mice_save_filepath=args[6]
            )

        # Merge datasets
        elif action == Codes.CLEANER_ACTIONS['merge']:
            args[0].merge_datasets(
                filenames=args[1], 
                extending_rows=args[2]
            )

        # Change dtype
        elif action == Codes.CLEANER_ACTIONS['set_dtype']:
            args[0].set_data_type(
                columns=args[1], 
                data_type=args[2]
            )

        # Write file
        elif action == Codes.CLEANER_ACTIONS['write_csv']:
            args[0].write_csv(filename=args[1]) 

    # Analysing
    elif stage == Codes.WORKFLOW_STAGE['analyse_data']:
        # Get stats
        if action == Codes.ANALYSER_ACTIONS['get_stats']:
            args[0].get_dataset_stats()

        # Summarise float column
        elif action == Codes.ANALYSER_ACTIONS['summarise_float_column']:
            args[0].summarise_float_column(
                column=args[1]
            )

        # K means
        elif action == Codes.ANALYSER_ACTIONS['kmeans']:
            args[0].k_means(
                k=args[1], 
                random_state=args[2], 
                max_iter=args[3]
            )

        # K means string summary
        elif action == Codes.ANALYSER_ACTIONS['kmeans_string_summary']:
            return(args[0].get_kmeans_summary_string())

        # Predict from k means
        elif action == Codes.ANALYSER_ACTIONS['kmeans_predict']:
            return(
                args[0].predict_from_kmeans(
                    data=args[1], 
                    update_model=args[2])
            )

        # Random forest
        elif action == Codes.ANALYSER_ACTIONS['random_forest']:
            args[0].random_forest(
                target_column=args[1], 
                subset_X=args[2],
                data_split=args[3], 
                random_state=args[4]
            ) 

        # Predict random forest
        elif action == Codes.ANALYSER_ACTIONS['predict_random_forest']:
            return(
                args[0].predict_random_forest(data=args[1]) 
            )

        # Neural network
        elif action == Codes.ANALYSER_ACTIONS['neural_network']:
            args[0].dense_nn_regression(
                target_column=args[1], 
                subset_X=args[2],
                data_split=args[3], 
                filename=args[4], 
                hist_filename=args[5],
                epochs=args[6], 
                random_state=args[7], 
                verbosity=args[8]
            )

        # Predict neural network
        elif action == Codes.ANALYSER_ACTIONS['predict_neural_network']:
            return(
                args[0].predict_from_nn(
                    data=args[1], 
                    target_column=args[2],
                    model_path=args[3])
            )

        # PCA
        elif action == Codes.ANALYSER_ACTIONS['pca']:
            args[0].pca_sk(
                subset_X=args[1], 
                normalise_data=args[2], 
                n_components=args[3],
                target_column=args[4], 
                save_transform_name=args[5]
            )

        # HCA
        elif action == Codes.ANALYSER_ACTIONS['hca']:
            args[0].hca(
                transpose=args[1],
                normalise=args[2], 
                subset_X=args[3]
            )

        # CC
        elif action == Codes.ANALYSER_ACTIONS['cc']:
            args[0].correlation_coefficients(
                subset_X=args[1]
            )
    
    # Plotting
    elif stage == Codes.WORKFLOW_STAGE['plot']:
        REV_PLOTTER_ACTIONS = Codes.get_reversed_dictionary(Codes.PLOTTER_ACTIONS)

        # Scatter plot of empty cells
        if action == Codes.PLOTTER_ACTIONS['visualise_empty_cells']:
            args[0].visualise_empty_cells(
                filename=args[1], 
                figsize=args[2], 
                label_font_size=args[3],
                title_font_size=args[4], 
                rotation=args[5], 
                colour=args[6]
            )

        # Bar plot of empty cells
        elif action == Codes.PLOTTER_ACTIONS['visualise_empty_cells_bar']:
            args[0].visualise_empty_cells_bar(
                filename=args[1], 
                figsize=args[2], 
                label_font_size=args[3],
                title_font_size=args[4], 
                rotation=args[5], 
                colour=args[6]
            )

        # Scatter plot of two dimensions of k means
        elif action == Codes.PLOTTER_ACTIONS['2d_kmeans']:
            args[0].plot_2d_comparison_kmeans(
                col1=args[1], 
                col2=args[2], 
                filename=args[3],
                plot_centres=args[4], 
                colours=args[5]
            )

        # Bar plot of RF importances
        elif action == Codes.PLOTTER_ACTIONS['rf_importances']:
            args[0].plot_rf_importances(
                filename=args[1], 
                ordered=args[2], 
                big_to_small=args[3], 
                horizontal=args[4], 
                figsize=args[5], 
                label_font_size=args[6],
                title_font_size=args[7], 
                rotation=args[8], 
                colour=args[9]
            )

        # Bar plot of PCA features
        elif action == Codes.PLOTTER_ACTIONS['pca_feature_bar']:
            args[0].plot_pca_feature_bar(
                filename=args[1], 
                figsize=args[2], 
                label_font_size=args[3], 
                title_font_size=args[4], 
                rotation=args[5], 
                colour=args[6]
            )

        # Cumulative sum of PCA variance explained
        elif action == Codes.PLOTTER_ACTIONS['pca_cumsum']:
            args[0].plot_pca_cumulative_importance(
                filename=args[1], 
                figsize=args[2], 
                label_font_size=args[3], 
                title_font_size=args[4], 
                rotation=args[5],
                colour=args[6]
            )

        # Dendrogram
        elif action == Codes.PLOTTER_ACTIONS['dendrogram']:
            args[0].plot_dendrogram(
                filename=args[1], 
                labels=args[2], 
                figsize=args[3], 
                label_font_size=args[4], 
                title_font_size=args[5], 
                rotation=args[6], 
                colour=args[7]
            ) 

        # CC Heatmaps
        elif action == Codes.PLOTTER_ACTIONS['cc_heatmap']:
            data = args[0].plot_correlations_heatmap(
                filenames=args[1], 
                figsize=args[2], 
                cmap=args[3]
            )
            for d in data:
                pf = ProcessFile()
                pf.process = process_obj
                pf.stage_name = REV_WORKFLOW_STAGE[stage]
                pf.stage_action_name = REV_PLOTTER_ACTIONS[action]
                pf.expected_filename = d['expected_filename']
                pf.file.save(
                    d['filename'], ContentFile(d['content'].getvalue()) #contentfile..getvalue()
                )


        # TF model losses plot
        elif action == Codes.PLOTTER_ACTIONS['tf_losses']:
            args[0].plot_tf_model_losses(
                hist_path=args[1], 
                filename=args[2]
            )

        # TF model predictions
        elif action == Codes.PLOTTER_ACTIONS['tf_predictions']:
            args[0].plot_model_predictions(
                predictions=args[1], 
                actual=args[2], 
                filename=args[3]
            ) 

        # TF model predictions histogram
        elif action == Codes.PLOTTER_ACTIONS['tf_predictions_hist']:
            args[0].bar_tf_model_predictions(
                predictions=args[1],
                actual=args[2], 
                filename=args[3]
            ) 
    
    # Making reports
    elif stage == Codes.WORKFLOW_STAGE['report']:
        
        # Data report
        if action == Codes.REPORT_ACTIONS['data_report']:
            args[0].make_data_report(
                datacleaner=args[1],
                filepath=args[2]
            )

        # Cleaning report
        elif action == Codes.REPORT_ACTIONS['cleaning_report']:
            args[0].make_cleaner_report(
                datacleaner=args[1], 
                filename=args[2]
            )

        # Analysis report
        elif action == Codes.REPORT_ACTIONS['analysis_report']:
            data = args[0].make_analysis_report(
                analyser=args[1], 
                filepath='analysis_report.pdf',
            )
            pf = ProcessFile()
            pf.process = process_obj
            pf.stage_name = 'report'
            pf.stage_action_name = 'analysis_report'
            pf.expected_filename = data['expected_filename']
            pf.file.save(
                data['filename'], ContentFile(data['content']) #content..getvalue()
            )


def easy_processor(process_obj, 
                index_col: int=None, 
                sheet_name: int=None,
                cleaner_lab: str=None,
                cleaner_unit: str=None,
                cleaner_action: str=None,
                cleaner_imputing_action: str=None,
                analyser_action: str=None,
                plotter_action: str=None,
                report_action: str=None):
    
    uploaded_filepath = process_obj.process_files.filter(stage_name='load_data').first().file.path
    uploaded_filename = os.path.basename(uploaded_filepath)
    is_csv = True if uploaded_filename.split('.')[-1]=="csv" else False
    is_excel = True if uploaded_filename.split('.')[-1]=="xlsx" else False
    is_xls = True if uploaded_filename.split('.')[-1]=="xls" else False

    # Loading data into a DataCleaner example
    args = {
        'filename': uploaded_filepath, # input file
        'lab': Codes.CLEANER_LAB.get(cleaner_lab), # ALS 1
        'unit': Codes.CLEANER_UNITS.get(cleaner_unit), # ppb 0
        'index_col': index_col, # int starting index 0
        'username': process_obj.project.name,
        'is_excel': True if is_excel else False,
        'is_xls': True if is_xls else False,
        'sheet_name': sheet_name # int starting index 0
        }
    dc = processor(stage=Codes.WORKFLOW_STAGE['load_data'], args=args)

    # Remove a column
    processor(stage=Codes.WORKFLOW_STAGE['clean_data'], 
            action=Codes.CLEANER_ACTIONS[cleaner_action], #remove_columns
            args=[dc, ['Au: ME-MS23']])  # TODO we should take column names from user

    # Get the correlation coefficients
    a = Analyser(dc)
    processor(stage=Codes.WORKFLOW_STAGE['analyse_data'], 
            action=Codes.ANALYSER_ACTIONS[analyser_action], #cc
            args=[a, None])

    # Make heatmaps
    plot = Plotter(a)
    processor(stage=Codes.WORKFLOW_STAGE['plot'], 
            action=Codes.PLOTTER_ACTIONS[plotter_action], #cc_heatmap
            args=[plot, ['Pearson.png', 'Kendall.png', 'Spearman.png'], (14,14),'copper_r'], # TODO I think I should take it from user, 'copper_r'. Ask data-sci
            process_obj=process_obj) 

    # Make analyser report
    reportmaker = ReportMaker(process_obj=process_obj) # pass temp file dir/path
    processor(stage=Codes.WORKFLOW_STAGE['report'],
            action=Codes.REPORT_ACTIONS['analysis_report'],  #'analysis_report'
            args=[reportmaker, a, 'analysis_report.pdf'],
            process_obj=process_obj)


