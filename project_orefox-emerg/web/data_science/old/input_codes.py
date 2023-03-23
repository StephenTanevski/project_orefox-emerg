

class Codes:

    # Codes for different stages of the workflow
    WORKFLOW_STAGE = {
        'Load Data': 0,
        'Clean Data': 1,
        'Analyse Data': 2,
        'Plot': 3,
        'Report': 4
    }

    # Codes for different labs
    CLEANER_LAB = {
        None: 0,
        'ALS': 1,
        'OSNACA': 2,
        'Petrosea': 3
    }

    # Cleaning actions
    CLEANER_ACTIONS = {
        'Imputing': 0
    }

    # Imputing actions
    CLEANER_IMPUTING_ACTIONS = {
        'zero': 0
    }

    ANALYSER_ACTIONS = {
        
    }


    @staticmethod
    def get_reversed_dictionary(dictionary: dict) -> dict:
        """
        Takes a dictionary and returns the same dictionary, with keys as
        values and values as keys.

        Args:
            dictionary: The dictionary to be reversed.
        """
        return  {value: key for key, value in dictionary.items()}