# Default imports
import pandas as pd

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# Your solution code here
def rf_rfe(df):
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    model = RandomForestClassifier()
    rfe = RFE(model, 17)
    rfe.fit(X,y)

    #return pd.DataFrame(np.vstack((X.columns,rfe.ranking_)))
    return X.columns.values[rfe.ranking_==1].tolist()
