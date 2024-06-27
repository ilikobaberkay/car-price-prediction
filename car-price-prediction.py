import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

data = pd.read_csv('cars.csv')

data.drop(columns=['Car_ID'], inplace=True)
data['Year'] = data['Year'].fillna(data['Year'].mean())
data['Kilometers_Driven'] = data['Kilometers_Driven'].fillna(data['Kilometers_Driven'].median())
data['Transmission'] = data['Transmission'].fillna(data['Transmission'].mode()[0])
data['Owner_Type'] = data['Owner_Type'].fillna(data['Owner_Type'].mode()[0])
data['Seats'] = data['Seats'].astype('int64')

le = LabelEncoder()
data['Transmission'] = le.fit_transform(data['Transmission'].values)
print(data.head())
data['Fuel_Type'] = le.fit_transform(data['Fuel_Type'].values)


ohe = OneHotEncoder()
owner_type_encoded = ohe.fit_transform(data.iloc[:, 6:7].values).toarray()
owner_type_encoded = pd.DataFrame(data=owner_type_encoded, columns=['First', 'Second', 'Third'])
owner_type_encoded.drop(columns=['Third'], inplace=True)

data.drop(columns=['Owner_Type'], inplace=True)
cc = pd.concat([data, owner_type_encoded], axis=1)
mean_prices = cc.groupby('Brand')['Price'].mean()
overall_median = mean_prices.median()
cc['Categorized Brands'] =  cc['Brand'].map(lambda brand: 1 if mean_prices[brand] > overall_median else 0)
mean_prices_model = cc.groupby('Model')['Price'].mean()
overall_median_model = mean_prices_model.median()

cc['Categorized Models'] =  cc['Model'].map(lambda Model: 1 if mean_prices_model[Model] > overall_median_model else 0)
cc.drop(columns=['Brand', 'Model'], inplace=True)
def detect_outliers_iqr(column):

    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (column < lower_bound) | (column > upper_bound)

outliers_indices = []
for column in cc.columns:
    outliers_indices.extend(cc.index[detect_outliers_iqr(cc[column])].tolist())

unique_outliers_indices = list(set(outliers_indices))
df_clean = cc.drop(unique_outliers_indices)
corrmat = df_clean.corr()
#print(corrmat[8:9])

corrs = df_clean.corr()['Price'].drop('Price')
vals = corrs[abs(corrs) >= 0.5].index.tolist()
above_threshold = np.abs(corrs) > 0.5
num_high_correlation_columns = above_threshold.sum(axis=0)

x = df_clean.copy()
x.drop(columns=['Year', 'Kilometers_Driven', 'Fuel_Type', 'Seats', 'First', 'Second', 'Price'], inplace=True)
y = df_clean.iloc[:, 8:9].values
y = pd.DataFrame(data=y, columns=['Price'])

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42,test_size=0.2)


li = LinearRegression()
li.fit(x_train, y_train)

p1 = li.predict(x_train)
p2 = li.predict(x_test)



mae1 = mean_absolute_error(y_train, p1)
r1 = r2_score(y_train, p1)

mae2 = mean_absolute_error(y_test, p2)
r2 = r2_score(y_test, p2)

print("TRAIN")
print(mae1)
print(r1)
print("TEST")
print(mae2)
print(r2)