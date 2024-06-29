# SMS/Email Spam Classifier

#### 🔗 *[SMS/Email Spam Classifier](https://ssc-sd.streamlit.app/)*

![Python](https://img.shields.io/badge/Python-3.10-fcba03) ![Frontend](https://img.shields.io/badge/Frontend-Streamlit-red) ![Libraries](https://img.shields.io/badge/Libraries-Numpy_|_Pandas_|_NLTK_|_Scikit_learn-orange) ![Naive Bayes Classifiers](https://img.shields.io/badge/Naive_Bayes_Classifiers-GaussianNB_|_MultinomialNB_|_BernoulliNB-blue) ![Support Vector Machines](https://img.shields.io/badge/Support_Vector_Machines-SVC-blue) ![Nearest Neighbors Classifiers](https://img.shields.io/badge/Nearest_Neighbors_Classifiers-KNeighborsClassifier-blue) ![Linear Classifiers](https://img.shields.io/badge/Linear_Classifiers-LogisticRegression-blue) ![Ensemble Classifiers](https://img.shields.io/badge/Ensemble_Classifiers-RandomForestClassifier_|_AdaBoostClassifier_|_BaggingClassifier-blue) ![Ensemble Classifiers](https://img.shields.io/badge/Ensemble_Classifiers-ExtraTreesClassifier_|_GradientBoostingClassifier_|_XGBClassifier-blue) 

## Overview
This project implements a machine learning model to classify SMS and email messages as either spam or non-spam (ham). It showcases proficiency in natural language processing (NLP), data preprocessing techniques, and model building using various algorithms. The classifier is deployed with a Streamlit frontend for easy interaction.

## Technologies Used
- **Frontend**: Streamlit
- **Libraries**: NLTK, scikit-learn
- **Classifiers**: GaussianNB, MultinomialNB, BernoulliNB, SVC, KNeighborsClassifier, DecisionTreeClassifier, LogisticRegression, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, XGBClassifier


## Skills Showcased
- **Data Cleaning and Preprocessing**: Techniques include lowercasing, tokenization, removal of special characters, stop words, punctuation, and stemming.
- **Exploratory Data Analysis (EDA)**: Understand data distribution and characteristics through statistical summaries and visualizations.
- **Model Building**: Implementation of multiple classifiers including Naive Bayes, SVM, Decision Tree, Random Forest, Logistic Regression, AdaBoost, Bagging, Extra Trees, Gradient Boosting, and XGBoost.
- **Evaluation Metrics**: Use of accuracy, precision, and confusion matrix for model evaluation.
- **Deployment**: Creation of a Streamlit frontend for user interaction, showcasing model predictions on new text inputs.

## Project Workflow
1. **Data Cleaning**:
   - Lowercasing
   - Tokenization
   - Special characters removal
   - Stop words and punctuation removal
   - Stemming

2. **EDA**:
   - Statistical summaries
   - Visualizations (word clouds, histograms)

3. **Data Preprocessing**:
   - Tokenization
   - Removal of special characters
   - Removal of stop words and punctuation
   - Stemming

4. **Model Building**:
   - Implemented classifiers:
     - Naive Bayes (Gaussian, Multinomial, Bernoulli)
     - SVM (Sigmoid kernel)
     - K-Nearest Neighbors
     - Decision Tree
     - Logistic Regression
     - Random Forest
     - AdaBoost
     - Bagging
     - Extra Trees
     - Gradient Boosting
     - XGBoost

5. **Evaluation**:
   - Accuracy scores
   - Confusion matrices
   - Precision scores

6. **Deployment**:
   - Streamlit frontend for interaction
   - Input text for predictions
   - Display of predicted class (spam or ham)


## How to run the project locally?

1. Clone or download this repository to your local machine.
2. `cd` into the cloned folder.
3. Install virtual environment python package using command:

   ```
   pip install virtualenv
   ```
4. Create a virtual environment using command:

   ```
   python3 -m venv [Enter Folder name]
   ```
5. Activate virtual environment using command:

   ```
   source [virtual environment name]/bin/activate
   ```
6. Install all the libraries mentioned in the [requirements.txt](https://github.com/soumadeep-dey/Movie-Recommendation-System/blob/main/requirements.txt) file with the command:

   ```
    pip install -r requirements.txt
   ```
7. Install ipykernel using command:

   ```
   pip install ipykernel
   ```
8. Create a kernel user using command:

   ```
   ipython kernel install --user --name=[Enter kernel_name]
   ```
9. Run the file `app.py` by executing the command:

   ```
   streamlit run app.py
   ```
10. The streamlit app will locally run on your browser using your default browser or run it manually in any browser using  the local url provided in your terminal as follows:

    ```
     You can now view your Streamlit app in your browser.

      Local URL: http://localhost:8501 (port number can be different) [copy and paste in any browser]
      Network URL: http://192.168.0.103:8501

    ```

Hurray! That's it. 🥳
