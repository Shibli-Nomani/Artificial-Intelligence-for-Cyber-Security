![image](https://github.com/Shibli-Nomani/Artificial-Intelligence-for-Cyber-Security/assets/101654553/3ece4c4a-e3b7-48a2-b2eb-6e5d4c4ec2d0)
<h1>🛡️ Defending Cyberspace: The Quest to Combat Spam with Machine Learning and AI 📧💥<h2>

  SMS and email spam detection 🛡️ is paramount in cybersecurity to shield users from phishing 🎣, malware 🦠, and fraud 💸 threats. Leveraging machine learning 🤖 and artificial intelligence 🧠, automated analysis of message content, sender behavior, and metadata enables accurate identification and filtering of spam, fortifying cybersecurity defenses and preserving user privacy and security.

<h1>💾 SMS Spam Collection</h1>
The SMS Spam Collection is a public set of SMS labeled messages that have been collected for mobile phone spam research.

dataset: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

<h1> 👇Use Dataset in Google Drive<h1>
get the sms Dataset for this project
datasets: https://drive.google.com/drive/folders/1Tb2eN400U6tq2SxGrxv8Va77Z-z-wn7R
  
<h1>📮 Spam Detection with Perceptrons</h1>
**😏Apache SpamAssassin** is an open-source spam filter renowned as the top choice for enterprise-level email protection. It employs a comprehensive scoring system and various heuristic and statistical tests, including Bayesian filtering and DNS blocklists, to identify and block spam emails effectively. With its flexible and easy-to-extend architecture, SpamAssassin seamlessly integrates into various email systems, offering robust protection against unsolicited bulk email.

**🧠 Perceptrons in Neural Network**

In neural networks, a perceptron 🧠 is a fundamental building block, acting as a single artificial neuron that processes input data and produces an output based on weighted connections and an activation function. It serves as the basic unit for information processing in neural networks, mimicking the functionality of biological neurons.
![image](https://github.com/Shibli-Nomani/Artificial-Intelligence-for-Cyber-Security/assets/101654553/9d4ee121-2a4b-44f5-90e1-ea0b034493d4)

**📬 Spam Filtering**

Spam filtering is like sorting through your mail: you want to separate the junk mail (spam) from the important stuff (ham). It's done by looking at different parts of an email, like who sent it and what it says. Classification means deciding if an email is junk (spam) or not (ham), usually using a computer program that learns from examples of both types of emails.

Spam filtering with machine learning: automatically distinguishing unwanted emails (spam) from legitimate ones (ham) based on patterns and characteristics.

![image](https://github.com/Shibli-Nomani/Artificial-Intelligence-for-Cyber-Security/assets/101654553/453cdab5-d699-45d9-bcdf-c95efbf7eb1b)

<h1>😡 Detection with SVM(Supervised Machine Learning)</h1>
Support Vector Machine (SVM) is a supervised learning algorithm used for classification and regression tasks. It aims to find the hyperplane that best separates data points into different classes while maximizing the margin between them. SVM is effective for both linearly separable and nonlinearly separable data, utilizing the kernel trick for the latter. Compared to the Perceptron, SVM generally offers better performance by maximizing the margin and handling complex decision boundaries more effectively.

**Takeway**

- Minimize the classification errors

- Maximize the margin and fits more data in a class

- Effective classes are not linearly separable in a hyperplane

- wider margin fewer classification errors and narrow margin risk of overfitting (small changes in these points can significantly affect the position of the hyperplane, more sensitive to noise and outliers)

- margin helps for best separation between classes

- n-dimension space (2D/3D)

![image](https://github.com/Shibli-Nomani/Artificial-Intelligence-for-Cyber-Security/assets/101654553/cab0b180-d8d1-4121-9794-f82373428812)

<h1>😡Spam Detection with Linear Regression model(Supervised Machine Learning)</h1>
**Linear Regression 📈:** is a statistical method to model the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data, used for predicting continuous outcomes.

For spam detection, Linear Regression can be applied to features extracted from email metadata, such as sender information, email length, and frequency of certain words, to predict the likelihood of an email being spam based on these features. However, Linear Regression might not be the most suitable model for this task as it's primarily designed for continuous target variables, whereas spam detection typically involves binary classification (ham or spam).

![image](https://github.com/Shibli-Nomani/Artificial-Intelligence-for-Cyber-Security/assets/101654553/d92c1d1e-9766-4451-8807-fed6a19797cd)

<h1>😡 Spam Detection with Logistic Regression model(Supervised Machine Learning)</h1>

**Logistic Regression: 📈** Statistical method used for binary classification by modeling the probability of a categorical outcome based on predictor variables.

For email spam detection, Logistic Regression can analyze features such as email metadata and content to predict the likelihood of an email being spam (1) or not spam (0) based on these features.

![image](https://github.com/Shibli-Nomani/Artificial-Intelligence-for-Cyber-Security/assets/101654553/26aff5cd-378d-4ef5-92a1-bfa0ca104ceb)


<h1>😡 Spam Detection with Decision Tree model(Supervised Machine Learning)</h1>

**Decision Tree: 🌳** Supervised learning algorithm that partitions data into subsets based on feature values, aiming to create a tree-like structure to predict the target variable's value for new data points.

🎃 Trees recursively split data based on true/false conditions, aiming to minimize impurity until reaching leaf nodes.

- Root Node: Represents the entire dataset and is the starting point for the tree. It contains the feature that best splits the data. This process is typically performed using a measure of impurity or information gain, such as Gini impurity or entropy.

- Branches: Represent the possible values of the chosen feature. Each branch leads to a child node.

- Leaf Nodes: Terminal nodes that represent the final outcome or prediction. They contain the predicted class label.

In spam detection, Decision Trees analyze email features recursively, making binary decisions at each node based on feature values (e.g., sender address, email content), eventually classifying emails as spam (🛑) or not spam (✉️).

![image](https://github.com/Shibli-Nomani/Artificial-Intelligence-for-Cyber-Security/assets/101654553/e616ce18-79c0-4e26-a042-6ecf79da38ef)

<h1>😡 Spam Message (NLP)Detection with Naive Bayes model(Supervised Machine Learning)</h1>

**Naive Bayes 📧✉️** is a Probabilistic classifier that assumes independence between features and predicts the probability of a class given input features based on Bayes' theorem.

![image](https://github.com/Shibli-Nomani/Artificial-Intelligence-for-Cyber-Security/assets/101654553/61d25836-e1d5-48ad-b657-2e44e4d823ca)

**📌Example:** Imagine you're a teacher trying to predict whether a student will pass or fail a test based on two features: hours studied and attendance. Naive Bayes assumes that these features are independent, so the probability of passing or failing is calculated separately for each feature. For instance, if historically students who study a lot tend to pass regardless of attendance, Naive Bayes would predict a higher probability of passing for a student who studied many hours, even if their attendance was low.

In spam detection, Naive Bayes analyzes the probability of an email being spam or not spam by considering the occurrence of certain words or features in the email content. For instance, it calculates the likelihood of an email being spam given the presence of words like "free" or "offer", utilizing probabilities to classify emails.

<h2> 💥 Types of Naive Bayes</h2>
- Guassian Naive Bayes: Normal/contineous distribution, can be used in income classification where value is contineous
- Multinomial Navie Bayes: Discrete Count Data, NLP (spam filtering / sentiment analysis)
-Bernoulli Naive Bayes: Use for Binary / Boolean data. Only two outcomes. (Text Classification, NLP)

  - - Spam Filtering 🎋
  - - Sentiment Analysis 🎋
  - - Medical Diagonise 🎋
  - - Image Recognation 🎋
  - - Fraud Detection 🎋
   
<h1>👊 Steps of Spam Messaging (NLP) Detection with Naive Bayes</h1>

1. Lemmatization
2. TfIdf
3. Apply Naive Bayes
4. Prediction

- NLP (Natural Language Processing): 🗣️🤖 Field of study focusing on the interaction between computers and human language, enabling machines to understand, interpret, and generate human language.
- NLTK (Natural Language Toolkit): 📚🗣️ A leading platform for building Python programs to work with human language data, providing easy-to-use interfaces to over 50 corpora and lexical resources.
- Tokenization: 🧩 Process of breaking text into smaller units (tokens) such as words or sentences, facilitating analysis and processing in NLP tasks.
- BOW (Bag of Words): 🎒 Representation of text data as a collection of word occurrences, disregarding grammar and word order, commonly used in document classification and information retrieval tasks in NLP.
- Lemmatization: 📖🔡 A linguistic process that reduces words to their base or dictionary form (lemmas), aiding in text normalization and improving the accuracy of natural language processing tasks.
- TF-IDF (Term Frequency-Inverse Document Frequency): 📊📉 A statistical measure that evaluates the importance of a word in a document relative to a collection of documents, by multiplying its frequency (TF) by the inverse document frequency (IDF). It's commonly used for text mining and information retrieval tasks.

# 👽 Summary
SMS and email spam detection is critical for safeguarding users against phishing attacks, malware distribution, and fraudulent activities. Machine learning and artificial intelligence techniques enable automated analysis of message content, sender behavior, and metadata to accurately identify and filter out spam, thereby enhancing cybersecurity defenses and protecting user privacy and security.
## Authors

- [@LinkedIn Khan MD Shibli Nomani](https://www.linkedin.com/in/khan-md-shibli-nomani-45445612b/)
