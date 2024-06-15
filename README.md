# S24-MachineLearning

### Assignment 1 Overview

In Assignment 1, I applied linear and polynomial regression techniques to a dataset, evaluating their performance using metrics such as Mean Squared Error (MSE) and R-squared (R²) score. I constructed a polynomial regression pipeline, tuned its degree using GridSearchCV, and compared the results to linear regression. The polynomial regression model, optimized through GridSearchCV, showed improved performance compared to the linear model.

Additionally, I analyzed feature correlations and identified a pair of highly correlated features, which could impact model performance. In the second part, I preprocessed a Pokémon dataset by handling missing values, encoding categorical variables, and scaling the features. I then compared Logistic Regression, K-Nearest Neighbors, and Gaussian Naive Bayes classifiers, with K-Nearest Neighbors yielding the best overall performance.


### Assignment 2.1 Overview

In Assignment 2, I tackled a classification problem where the goal was to identify a specific user, "Waldo," from a large dataset. To achieve this, I conducted a comprehensive Exploratory Data Analysis (EDA) to understand the data's structure, imbalances, and key features. The analysis revealed that certain features like locale, location, and time-related characteristics were crucial for distinguishing Waldo, while other features like the operating system and frequently visited websites were less significant than initially expected.

I utilized various machine learning techniques, including feature importance analysis and model evaluation, to refine the prediction process. Technologies such as scikit-learn for model training, feature importance estimation, and precision-recall metrics were employed. The final model, based on the significant features identified through EDA and feature importance analysis, was evaluated for performance and used to make predictions on a verification dataset. 

Throughout this assignment, I enhanced my skills in feature engineering, imbalanced data handling, and model evaluation.

### Assignment 2.2 Overview

In Assignment 2, I focused on image classification using the CIFAR-10 dataset, aiming to develop and evaluate two distinct convolutional neural network (CNN) models: a custom CNN and a transfer learning model based on ResNet-50. I started by building and training a custom CNN model, incorporating various techniques such as batch normalization and dropout to enhance its performance. This model achieved a ROC-AUC score of 94.40% and an accuracy of 65.67%, with a relatively fast inference speed of 0.0128 ms per image.

For the second part, I employed transfer learning with ResNet-50, leveraging a pre-trained model to classify the CIFAR-10 images. This approach led to a significantly higher accuracy of 76.46% and a ROC-AUC score of 96.66%, though with a larger model size and slower inference speed of 0.0668 ms per image.

Through this assignment, I deepened my understanding of CNN architectures, model optimization, and the trade-offs between custom and pre-trained models. I learned how transfer learning can greatly enhance performance but may come with increased complexity and resource requirements, while custom models offer more control and faster inference at the cost of potentially lower accuracy.

### Assignment Bonus Overview

In the Bonus Assignment, I evaluated multiple machine learning models on the CIFAR-10 dataset, including a baseline custom CNN, a self-supervised autoencoder, and models using auxiliary and ensemble learning techniques. The baseline model, trained on 50,000 labeled images, achieved the highest accuracy of 73.52% with a fast inference time of 0.3418 ms per image.

The self-supervised autoencoder, which used a combination of 5,000 labeled and 50,000 unlabeled images, had a lower accuracy of 35.16% and a longer inference time of 1.0316 ms per image. The auxiliary learning model, trained with both labeled and unlabeled data, reached an accuracy of 44.90% but had a significantly larger model size and slower inference time.

The ensemble model, which integrated multiple approaches, offered a balanced performance with 65.05% accuracy. This comprehensive evaluation highlighted the trade-offs between model accuracy, complexity, and inference efficiency, underscoring the importance of selecting the right approach based on specific needs and constraints.