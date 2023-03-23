from src.cleaner import DataCleaner
from src.resources.input_codes import Codes

def processor(stage: int,data_filename: str, action: int=None, args: list=None, 
        lab: int='ALS', cleaner=None):
    """A function that takes input and calls the necessary function.

    Args:
        stage: The stage of the workflow the user is at. Refer to
          input_codes.WORKFLOW_STAGE for details.
        action: The numerical id of the action to take. These are not unique,
          but with reference to the stage will be unique.
        args: A list of parameters to define how the function should be called.
        lab: The lab to use in the loading data stage. Refer to
          input_codes.CLEANER_LAB for details. # TODO Also an arg maybe?
        data_filename: The filename for the dataset to load. # TODO Make this an arg?
        cleaner: The DataCleaner to use for cleaning stage.

    Returns:
        DataCleaner: In the case of loading and cleaning stages.
    """

    # Loading data in initially
    if stage == Codes.WORKFLOW_STAGE['Load Data']:
        cleaner = DataCleaner(
            filename=data_filename, 
            lab=Codes.CLEANER_LAB[lab]
        )
        # Keep adding params here

        return cleaner

    # Cleaning
    elif stage == Codes.WORKFLOW_STAGE['Clean Data']:

        # Imputing values
        if action == Codes.CLEANER_ACTIONS['Imputing']:
            cleaner.replace_empty_entries()

    # Analysing
    elif stage == Codes.WORKFLOW_STAGE['Analyse Data']:
        pass # TODO
    
    # Plotting
    elif stage == Codes.WORKFLOW_STAGE['Plot']:
        pass # TODO
    
    # Making reports
    elif stage == Codes.WORKFLOW_STAGE['Report']:
        pass # TODO



if __name__ == '__main__':
    processor(stage=0, data_filename="test/data/ALS/BR20279856.csv")