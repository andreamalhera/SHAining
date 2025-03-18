import os
import pandas as pd

from shaining.shapley import ShapleyTask

def test_shapley():
    INPUT_PARAMS = {
        "pipeline_step": "shapley_computation",
        'input_path': 'data/test/shapley_test/shapley_test.csv',
        'output_path': 'output',
        'feature_names': ["aq1", "eav", "ekbr3"],
        'miners': ["sm1"]
    }
    VALIDATION_RESULT = {'algorithm': {0: 'sm1', 1: 'sm1', 2: 'sm1', 3: 'sm1', 4: 'sm1', 5: 'sm1'},
                         'metric': {0: 'fitness', 1: 'fitness', 2: 'fitness', 3: 'precision', 4: 'precision', 5: 'precision'},
                         'feature_name': {0: 'aq1', 1: 'eav', 2: 'ekbr3', 3: 'aq1', 4: 'eav', 5: 'ekbr3'},
                         'feature_value': {0: '100', 1: '20', 2: '450', 3: '100', 4: '20', 5: '450'},
                         'shapley_value': {0: 0.3275753333333333, 1: 0.2821708333333333, 2: 0.2476898333333333, 3: -0.034585, 4: 0.42396900000000004, 5: 0.30467099999999997}}

    # Creating data for the test
    df = pd.DataFrame({'log':['genEL1_100_20_450','genEL2_100_20_nan','genEL3_100_nan_450','genEL4_nan_20_450','genEL5_100_nan_nan','genEL6_nan_20_nan','genEL7_nan_nan_450'],
                       'fitness_sm1':[0.857436,0.785,0.920503,0.835341,0.888112,0.882465,0.678],
                       'precision_sm1':[0.694055,0.783,0.631823,0.903646,0.336134,0.981419,0.894]})
    save_path = os.path.join('data/test/shapley_test','shapley_test.csv')
    os.makedirs(os.path.join('data/test/shapley_test'), exist_ok=True)
    df.to_csv(save_path,index=False)

    # Calculating the shapely values
    shapley = ShapleyTask(params=INPUT_PARAMS)
    shapely_value = shapley.results.to_dict()

    # Deleting the test data
    os.remove(save_path)
    assert shapely_value == VALIDATION_RESULT