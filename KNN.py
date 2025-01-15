import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter

#%%
data_file = 'arrhythmia.data'
df = pd.read_csv(data_file, header=None)

class_counts = Counter(df.iloc[:,-1])
classes_to_augment = [class_label for class_label, count,  in class_counts.items() if count<6]
for class_label in classes_to_augment:
    class_rows = df[df.iloc[:,-1]==class_label]
    while class_counts[class_label]<6:
        df = pd.concat([df, class_rows], ignore_index=True)
        class_counts[class_label] += len(class_rows)
class_counts = Counter(df.iloc[:,-1])
for class_label, count in class_counts.items():
    print(f" Class {class_label} has {count}")
#%%
df.replace("?", np.nan, inplace=True)

# Continue with your imputation as before
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputed_data = imputer.fit_transform(df)
df_imputed = pd.DataFrame(imputed_data, columns=df.columns)
print(df_imputed.shape)

#%%
X = df_imputed.iloc[ :,:-1].values
y=df_imputed.iloc[:,-1]
print(np.shape(X))
print(np.shape(y))
#%%
X_res, y_res = SMOTE(random_state=0).fit_resample(X, y)
print(X_res.shape)
print(y_res.shape)

#%%
sc = StandardScaler()
X_scaled = sc.fit_transform(X_res)
pca = PCA()
pca.fit(X_scaled)
pca_data = pca.transform(X_scaled)

#%%
per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
labels = ['PC' + str(i) for i in range(1, len(per_var) + 1)]
plt.bar(x=range(1,len(per_var) + 1), height=per_var, tick_label=labels)
plt.xlabel('Principal Component')
plt.ylabel(' Percentage of Explained Variance')
plt.title('Cree Plot')
plt.show()

#%%
threshold = 1  
important_components = per_var[per_var >= threshold]

# Adjust the plot to only show significant components based on the threshold
plt.bar(x=range(1, len(important_components) + 1), height=important_components, tick_label=labels[:len(important_components)])
plt.xlabel('Principal Component')
plt.ylabel('Percentage of Explained Variance')
plt.title('Filtered Scree Plot')
plt.show()

#%%

important_indices = [i for i, variance in enumerate(per_var) if variance >= threshold]
important_pca_data = pca_data[:, important_indices]
print(important_pca_data.shape)

#%%

new_X = pd.DataFrame(important_pca_data)
print(new_X.shape)
print(y_res.shape)

#%%
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(new_X, y_res, test_size=0.2, random_state=42)

classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)
# Predictions and evaluation
y_pred = classifier.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

#%%

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
def plot(y_true, y_pred):
    labels = unique_labels(y_test)
    columns = [f'Pred:{label}' for label in labels ]
    index = [f'True{label}' for label in labels ]
    table = pd.DataFrame(confusion_matrix(y_true, y_pred), columns=columns, index=index)
    return table

plot(y_pred, y_test)

#%%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Assuming y_test and y_pred are defined
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 8))  # Set a larger figure size
sns.heatmap(cm, cmap="Greens", annot=True, fmt="d", linewidths=.5, cbar=True)
plt.title("Confusion Matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()




