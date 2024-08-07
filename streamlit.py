import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

def preprocess_data(data):
    data['TB/U'] = data['TB/U'].apply(lambda x: 1 if x == 'Pendek' else 0)

    # Select relevant columns
    X = data[['Berat', 'Usia Saat Ukur (Bulan)', 'ZS TB/U']]
    y = data['TB/U']

    # Handle missing values
    # imputer = SimpleImputer(strategy='mean')
    # X = imputer.fit_transform(X)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

def visualize_data(X, y):
    st.subheader('Scatter Plots')
    df = pd.DataFrame(X, columns=['Berat', 'Usia Saat Ukur (Bulan)', 'ZS TB/U'])
    df['Stunting'] = y

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    sns.scatterplot(data=df, x='Berat', y='ZS TB/U', hue='Stunting', ax=axs[0])
    axs[0].set_title('Z-score TB/U vs. Weight (Berat)')
    axs[0].set_xlabel('Weight (Berat)')
    axs[0].set_ylabel('Z-score TB/U')

    sns.scatterplot(data=df, x='Usia Saat Ukur (Bulan)', y='ZS TB/U', hue='Stunting', ax=axs[1])
    axs[1].set_title('Z-score TB/U vs. Age in months')
    axs[1].set_xlabel('Age in months')
    axs[1].set_ylabel('Z-score TB/U')

    st.pyplot(fig)

def build_and_train_lstm(X_train, y_train, X_val, y_val):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))  # Adding Dropout to prevent overfitting
    model.add(Dense(1, activation='sigmoid'))  # Binary classification

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    
    return model, history

def plot_training_history(history):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    
    axs[0].plot(history.history['accuracy'], label='train accuracy')
    axs[0].plot(history.history['val_accuracy'], label='val accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()
    axs[0].set_title('Training and Validation Accuracy')
    
    axs[1].plot(history.history['loss'], label='train loss')
    axs[1].plot(history.history['val_loss'], label='val loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].set_title('Training and Validation Loss')
    
    st.pyplot(fig)

# Streamlit app
st.title('Klasifikasi dan Visualisasi Stunting')

# Sidebar menu
menu = st.sidebar.selectbox('Menu', ['Classification', 'Visualization'])

# File uploader
uploaded_file = st.sidebar.file_uploader('Upload your Excel file', type=['xlsx'])

if uploaded_file:
    data = load_data(uploaded_file)
    X, y = preprocess_data(data)

    if menu == 'Classification':
        st.header('Classification')

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42
        )

        # Reshape data for LSTM
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

        model, history = build_and_train_lstm(X_train, y_train, X_val, y_val)

        st.subheader('Training History')
        st.write('Berikut grafik loss dan accuracy dari data stunting dengan 50 epoch ')
        plot_training_history(history)

    elif menu == 'Visualization':
        st.header('Visualization Data')
        st.write('Berikut visualisasi dari data stunting dengan menggunakan scatter plot')
        visualize_data(X, y)
else:
    st.write('Please upload an Excel file to proceed.')
