## Problem statement is based on the some feature predict the customer they are existed or not
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow
import datetime
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
## load the data set

data = pd.read_csv("Churn_Modelling.csv")


## Preprocess the data
### drop irrelevant columns
data = data.drop(["RowNumber","CustomerId","Surname"],axis=1)


label_encoder_gender = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data["Gender"])
# print (data.head())

onehot_encoder_geom = OneHotEncoder()
geo_encoder_encoder=onehot_encoder_geom.fit_transform(data[["Geography"]]).toarray()

geo_encoder_df = pd.DataFrame(geo_encoder_encoder,columns=onehot_encoder_geom.get_feature_names_out(['Geography']))
data=pd.concat([data.drop('Geography',axis=1),geo_encoder_df],axis=1)



## Save the encoders and sscaler
with open('label_encoder_gender.pkl','wb') as file:
    pickle.dump(label_encoder_gender,file)

with open('onehot_encoder_geom.pkl','wb') as file:
    pickle.dump(onehot_encoder_geom,file)



## DiVide the dataset into indepent and dependent features
X=data.drop('Exited',axis=1)
y=data['Exited']

## Split the data in training and tetsing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

## Scale these features
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

# print(X_train)

with open('scaler.pkl','wb') as file:
    pickle.dump(scaler,file)

#ANN Implementation
## it tells the how many columns present in the dataset

# print(X_train.shape[1],)


## Build Our ANN Model
model=Sequential([
    Dense(64,activation='relu',input_shape=(X_train.shape[1],)), ## HL1 Connected wwith input layer
    Dense(32,activation='relu'), ## HL2
    Dense(1,activation='sigmoid')  ## output layer
]

)

# print(model.summary())

opt=tensorflow.keras.optimizers.Adam(learning_rate=0.01)
loss=tensorflow.keras.losses.BinaryCrossentropy()

model.compile(optimizer=opt,loss="binary_crossentropy",metrics=['accuracy'])

## Set up the Tensorboard


log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)

## Set up Early Stopping
early_stopping_callback=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)

### Train the model
history=model.fit(
    X_train,y_train,validation_data=(X_test,y_test),epochs=100,
    callbacks=[tensorflow_callback,early_stopping_callback]
)

model.save('model.h5') 