# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.preprocessing import LabelEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.utils import class_weight
# from imblearn.over_sampling import SMOTE
# import pickle
# import matplotlib.pyplot as plt

# # Function to preprocess the data
# def preprocess_data(df):
#     # Fill missing values
#     imputer = SimpleImputer(strategy='most_frequent')
#     df = df.copy()
#     df.fillna(df.mode().iloc[0], inplace=True)
    
#     # Encode categorical features
#     label_encoders = {}
#     for column in df.select_dtypes(include=['object']).columns:
#         le = LabelEncoder()
#         df[column] = le.fit_transform(df[column])
#         label_encoders[column] = le
    
#     return df, label_encoders

# # Function to train and evaluate the models
# def train_models(df):
#     # Ensure 'left' is in the dataset
#     if 'left' not in df.columns:
#         st.error("The dataset does not contain a 'left' column.")
#         return None, None, None, None

#     # Prepare data
#     X = df.drop('left', axis=1)
#     y = df['left']
    
#     # Handle class imbalance
#     smote = SMOTE(random_state=42)
#     X, y = smote.fit_resample(X, y)
    
#     # Split data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
#     # Train models
#     models = {
#         'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
#         'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
#     }
#     results = {}
    
#     for name, model in models.items():
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         report = classification_report(y_test, y_pred, output_dict=True)
#         results[name] = (accuracy, report, model)
    
#     return results

# # Function to display feature importance
# def display_feature_importance(model, X):
#     importances = model.feature_importances_
#     features = X.columns
#     sorted_indices = importances.argsort()[::-1]
    
#     fig, ax = plt.subplots()
#     ax.barh(features[sorted_indices], importances[sorted_indices])
#     ax.set_title('Feature Importance')
#     ax.set_xlabel('Importance')
    
#     st.pyplot(fig)

# def main():
#     st.title('Employee Data Dashboard with ML')

#     # File uploader in the sidebar
#     uploaded_file = st.sidebar.file_uploader("Choose a file")
#     if uploaded_file is not None:
#         # Read the uploaded file
#         df = pd.read_csv(uploaded_file)  # Adjust if the file is an Excel file
        
#         # Display the column names for verification
#         st.write("Column names in the uploaded file:", df.columns.tolist())

#         # Preprocess the data
#         df_processed, label_encoders = preprocess_data(df)

#         # Create tabs
#         tab1, tab2 = st.tabs(["Visualizations", "Model Training"])

#         # Tab 1: Visualizations
#         with tab1:
#             st.header("Visualizations")
            
#             # Initialize filter variables
#             filtered_df = df_processed
#             column_names = df_processed.columns.tolist()
            
#             # Initial filtering options in the sidebar
#             if st.sidebar.checkbox('Show Initial Filter Options'):
#                 selected_column = st.sidebar.selectbox('Select a column to filter by', column_names)
#                 unique_values = df_processed[selected_column].unique()
#                 selected_value = st.sidebar.selectbox(f'Select a value from {selected_column}', unique_values)
                
#                 # Filter the dataframe based on selected value and column
#                 filtered_df = df_processed[df_processed[selected_column] == selected_value]

#             # Display the filtered dataframe within an expander
#             with st.expander("View Filtered Data"):
#                 st.dataframe(filtered_df)
            
#             # Visualization Section
#             if st.checkbox('Show Data Distribution'):
#                 # Plot distribution of a selected column
#                 col_to_plot = st.selectbox('Select a column to plot distribution', column_names)
#                 fig = px.histogram(filtered_df, x=col_to_plot, title=f'Distribution of {col_to_plot}')
#                 st.plotly_chart(fig)

#             # Visualization of counts
#             if st.checkbox('Show Value Counts'):
#                 # Select a column to show value counts
#                 count_column = st.selectbox('Select a column to show value counts', column_names)
#                 value_counts = filtered_df[count_column].value_counts().reset_index()
#                 value_counts.columns = [count_column, 'Count']
                
#                 # Plot the value counts
#                 fig = px.bar(value_counts, x=count_column, y='Count', title=f'Value Counts of {count_column}')
#                 st.plotly_chart(fig)

#             # Correlation matrix
#             if st.checkbox('Show Correlation Matrix'):
#                 correlation_matrix = filtered_df.corr()
#                 st.write("Correlation Matrix:")
#                 st.dataframe(correlation_matrix)
                
#                 # Plot the correlation matrix
#                 fig = px.imshow(correlation_matrix, text_auto=True, title='Correlation Matrix')
#                 st.plotly_chart(fig)

#         # Tab 2: Model Training
#         with tab2:
#             st.header("Model Training and Evaluation")

#             # Train and evaluate the models
#             results = train_models(df_processed)
            
#             if results:
#                 for model_name, (accuracy, report, model) in results.items():
#                     st.write(f"Model: {model_name}")
#                     st.write(f"Accuracy: {accuracy:.2f}")
#                     st.write("Classification Report:")
#                     st.json(report)  # Display the classification report in JSON format

#                     # Display feature importance for Random Forest
#                     if model_name == 'Random Forest':
#                         display_feature_importance(model, df_processed.drop('left', axis=1))

#                     # Save the model
#                     model_filename = f'{model_name.replace(" ", "_").lower()}_model.pkl'
#                     with open(model_filename, 'wb') as file:
#                         pickle.dump(model, file)
#                     st.success(f"Model '{model_name}' trained and saved successfully!")

#                     # Provide download link for the model file
#                     st.download_button(
#                         label=f"Download {model_name} Model",
#                         data=open(model_filename, 'rb').read(),
#                         file_name=model_filename,
#                         mime='application/octet-stream'
#                     )

#             # Interactive prediction
#             st.subheader("Predict New Data")
#             input_data = {
#                 "satisfaction_level": st.number_input("Satisfaction Level", min_value=0.0, max_value=1.0, value=0.5),
#                 "last_evaluation": st.number_input("Last Evaluation", min_value=0.0, max_value=1.0, value=0.5),
#                 "number_project": st.number_input("Number of Projects", min_value=0, max_value=10, value=3),
#                 "average_montly_hours": st.number_input("Average Monthly Hours", min_value=0, max_value=500, value=200),
#                 "time_spend_company": st.number_input("Time Spent in Company", min_value=0, max_value=10, value=3),
#                 "Work_accident": st.selectbox("Work Accident", [0, 1]),
#                 "promotion_last_5years": st.selectbox("Promotion in Last 5 Years", [0, 1]),
#                 "sales": st.selectbox("Sales Department", df['sales'].unique()),
#                 "salary": st.selectbox("Salary", df['salary'].unique())
#             }

#             if st.button("Predict"):
#                 input_df = pd.DataFrame([input_data])
#                 input_df, _ = preprocess_data(input_df)
#                 model = results['Random Forest'][2]  # Use the RandomForest model for prediction
#                 prediction = model.predict(input_df)
#                 st.write("Predicted Outcome: ", "Left" if prediction[0] == 1 else "Stayed")
    
# if __name__ == '__main__':
#     main()




import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import pickle
import matplotlib.pyplot as plt

# Function to preprocess the data
def preprocess_data(df):
    # Fill missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df = df.copy()
    df.fillna(df.mode().iloc[0], inplace=True)
    
    # Encode categorical features
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    return df, label_encoders

# Function to train and evaluate the models
def train_models(df):
    # Ensure 'left' is in the dataset
    if 'left' not in df.columns:
        st.error("The dataset does not contain a 'left' column.")
        return None, None, None, None

    # Prepare data
    X = df.drop('left', axis=1)
    y = df['left']
    
    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = (accuracy, report, model)
    
    return results

# Function to display feature importance
def display_feature_importance(model, X):
    importances = model.feature_importances_
    features = X.columns
    sorted_indices = importances.argsort()[::-1]
    
    fig, ax = plt.subplots()
    ax.barh(features[sorted_indices], importances[sorted_indices])
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    
    st.pyplot(fig)

def main():
    st.title('Employee Data Dashboard with ML')

    # File uploader in the sidebar
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)  # Adjust if the file is an Excel file
        
        # Display the column names for verification
        st.write("Column names in the uploaded file:", df.columns.tolist())

        # Preprocess the data
        df_processed, label_encoders = preprocess_data(df)
        
        # Sidebar options
        show_original_data = st.sidebar.checkbox('Show Original Data')
        
        if show_original_data:
            st.subheader("Original Data")
            st.dataframe(df)

        # Create tabs
        tab1, tab2 = st.tabs(["Visualizations", "Model Training"])

        # Tab 1: Visualizations
        with tab1:
            st.header("Visualizations")
            
            # Initialize filter variables
            filtered_df = df_processed
            column_names = df_processed.columns.tolist()
            
            # Filtering options in the sidebar
            if st.sidebar.checkbox('Show Filtering Options'):
                selected_column = st.sidebar.selectbox('Select a column to filter by', column_names)
                unique_values = df_processed[selected_column].unique()
                selected_value = st.sidebar.selectbox(f'Select a value from {selected_column}', unique_values)
                
                # Filter the dataframe based on selected value and column
                filtered_df = df_processed[df_processed[selected_column] == selected_value]

            # Display the filtered dataframe within an expander
            with st.expander("View Filtered Data"):
                st.dataframe(filtered_df)
            
            # Visualization Section
            if st.checkbox('Show Data Distribution'):
                # Plot distribution of a selected column
                col_to_plot = st.selectbox('Select a column to plot distribution', column_names)
                fig = px.histogram(filtered_df, x=col_to_plot, title=f'Distribution of {col_to_plot}')
                st.plotly_chart(fig)

            # Visualization of counts
            if st.checkbox('Show Value Counts'):
                # Select a column to show value counts
                count_column = st.selectbox('Select a column to show value counts', column_names)
                value_counts = filtered_df[count_column].value_counts().reset_index()
                value_counts.columns = [count_column, 'Count']
                
                # Plot the value counts
                fig = px.bar(value_counts, x=count_column, y='Count', title=f'Value Counts of {count_column}')
                st.plotly_chart(fig)

            # Correlation matrix
            if st.checkbox('Show Correlation Matrix'):
                correlation_matrix = filtered_df.corr()
                st.write("Correlation Matrix:")
                st.dataframe(correlation_matrix)
                
                # Plot the correlation matrix
                fig = px.imshow(correlation_matrix, text_auto=True, title='Correlation Matrix')
                st.plotly_chart(fig)

        # Tab 2: Model Training
        with tab2:
            st.header("Model Training and Evaluation")

            # Train and evaluate the models
            results = train_models(df_processed)
            
            if results:
                for model_name, (accuracy, report, model) in results.items():
                    st.write(f"Model: {model_name}")
                    st.write(f"Accuracy: {accuracy:.2f}")
                    st.write("Classification Report:")
                    st.json(report)  # Display the classification report in JSON format

                    # Display feature importance for Random Forest
                    if model_name == 'Random Forest':
                        display_feature_importance(model, df_processed.drop('left', axis=1))

                    # Save the model
                    model_filename = f'{model_name.replace(" ", "_").lower()}_model.pkl'
                    with open(model_filename, 'wb') as file:
                        pickle.dump(model, file)
                    st.success(f"Model '{model_name}' trained and saved successfully!")

                    # Provide download link for the model file
                    st.download_button(
                        label=f"Download {model_name} Model",
                        data=open(model_filename, 'rb').read(),
                        file_name=model_filename,
                        mime='application/octet-stream'
                    )

            # Interactive prediction
            st.subheader("Predict New Data")
            input_data = {
                "satisfaction_level": st.number_input("Satisfaction Level", min_value=0.0, max_value=1.0, value=0.5),
                "last_evaluation": st.number_input("Last Evaluation", min_value=0.0, max_value=1.0, value=0.5),
                "number_project": st.number_input("Number of Projects", min_value=0, max_value=10, value=3),
                "average_montly_hours": st.number_input("Average Monthly Hours", min_value=0, max_value=500, value=200),
                "time_spend_company": st.number_input("Time Spent in Company", min_value=0, max_value=10, value=3),
                "Work_accident": st.selectbox("Work Accident", [0, 1]),
                "promotion_last_5years": st.selectbox("Promotion in Last 5 Years", [0, 1]),
                "sales": st.selectbox("Sales Department", df['sales'].unique()),
                "salary": st.selectbox("Salary", df['salary'].unique())
            }

            if st.button("Predict"):
                input_df = pd.DataFrame([input_data])
                input_df, _ = preprocess_data(input_df)
                model = results['Random Forest'][2]  # Use the RandomForest model for prediction
                prediction = model.predict(input_df)
                st.write("Predicted Outcome: ", "Left" if prediction[0] == 1 else "Stayed")
    
if __name__ == '__main__':
    main()
