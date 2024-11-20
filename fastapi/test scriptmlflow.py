import mlflow
import pandas as pd

import data

logged_model = 'runs:/e8112bbf13104928ad44ac387ae496cb/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.

loaded_model.predict(pd.DataFrame(data))