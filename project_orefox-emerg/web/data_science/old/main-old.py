from . import cleaner


def processor(process, file_path):
    if process == "ALS":
        file_name = "OreFox_ALS.csv"
        dc = cleaner.DataCleaner(file_path, lab='ALS')
        # dc.write_csv('results_ALS.csv')
        content = dc.write_csv()
        return {'content': content, "file_name":file_name}
    
    elif process == 'OSNACA':
        file_name = "OreFox_OSNACA.csv"
        dc = cleaner.DataCleaner(file_path, lab='OSNACA')
        # dc.write_csv('results_OSNACA.csv')
        content = dc.write_csv()
        return {'content': content, "file_name": file_name}

