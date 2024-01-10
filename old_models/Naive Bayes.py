import numpy as np
import os
import glob
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import pickle

# Set the path to your directory containing text files
directory_path = "face40_details_new"

# Use glob to get a list of all text files in the directory
text_files = glob.glob(os.path.join(directory_path, "*.txt"))

# Limit to the first 5000 files
first_20000_files = text_files[:20000]

# Lists to store text content and corresponding labels
texts = []
labels = []

CelebA_Categories = {
    '5_o_Clock_Shadow': ['bristles', 'birstly stubble', 'stubble', '5 o clock shadow', '5 oh clock shadow'],
    'Arched_Eyebrows': ['Arched Eyebrows'],
    'Attractive': ['alluring', 'beautiful', 'charming', 'fair', 'good looking', 'gorgeous', 'handsome', 'lovely', 'adorable', 'pretty', 'seductive', 'mesmerising', 'bewitching'],
    'Bags_Under_Eyes': ['eye bags', 'bags under eyes'],
    'Bald': ['bald', 'no hair'],
    'Bangs': ['bangs', 'short hair', 'crew cut', 'bob'],
    'Big_Lips': ['big lips', 'large lips'],
    'Big_Nose': ['bulbous', 'big nose', 'large', 'huge', 'big nose'],
    'Black_Hair': ['black', 'dark'],
    'Blond_Hair': ['blonde','blond'],
    'Blurry': ['hazy', 'blurry'],
    'Brown_Hair': ['brown'],
    'Bushy_Eyebrows': ['bushy', 'big eyebrows', 'thick', 'big eyebrows'],
    'Chubby': ['fat', 'chunky', 'flabby', 'plump', 'stout', 'pudgy', 'big'],
    'Double_Chin': ['double chin'],
    'Eyeglasses': ['glasses', 'spectacles'],
    'Goatee': ['goatee'],
    'Gray_Hair': ['gray'],
    'Heavy_Makeup': ['makeup', 'make up'],
    'High_Cheekbones': ['high cheekbones','cheekbones'],
    'Male': ['male', 'he','man','He','Male','Man'],
    'Mouth_Slightly_Open': [],
    'Mustache': ['mustache'],
    'Narrow_Eyes': ['narrow eyes'],
    'No_Beard': ['no beard', 'clean shaven', 'shaved', 'clean shaved'],
    'Oval_Face': ['oval face'],
    'Pale_Skin': ['pale', 'pale skin', 'white'],
    'Pointy_Nose': ['pointy'],
    'Receding_Hairline': ['receding hairline', 'receding'],
    'Rosy_Cheeks': ['rosy cheeks'],
    'Sideburns': ['sideburns'],
    'Smiling': ['smiling'],
    'Straight_Hair': ['straight', 'straight hair'],
    'Wavy_Hair': ['wavy hair', 'wavy', 'curly', 'curly hair'],
    'Wearing_Earrings': ['ear rings', 'earrings'],
    'Wearing_Hat': ['hat', 'cap', 'fedora', 'beanie'],
    'Wearing_Lipstick': ['lipstick'],
    'Wearing_Necklace': ['necklace'],
    'Wearing_Necktie': ['necktie', 'neck tie'],
    'Young': ['young', 'youthful', 'juvenile', 'junior', 'teenage', 'teenager']
}

# Annotating the dataset
# Loop through the first 20000 files and read their contents
for file_path in first_20000_files:
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            words = line.split()
            # Remove punctuation and newline characters, and convert to lowercase
            words = [word.translate(str.maketrans("", "", string.punctuation)).lower() for word in words]
            words = [word.strip() for word in words if word]  # Remove empty strings
            for i in range(len(words)): 
                label_found = False
                
                # single word
                for category, keywords in CelebA_Categories.items():
                    if any(keyword == words[i] for keyword in keywords): # Once a category is found, break the inner loop
                        texts.append(words[i])
                        labels.append(category)
                        label_found = True
                        break

                # two words
                if i < len(words) - 1 and not label_found:
                    two_words = ' '.join(words[i:i + 2])
                    for category, keywords in CelebA_Categories.items():
                        if any(keyword == two_words for keyword in keywords): # Once a category is found, break the inner loop
                            texts.append(two_words)
                            labels.append(category)
                            label_found = True
                            break  
                
                # If no label is found, assign "Other"
                if not label_found:
                    texts.append(words[i])
                    labels.append('Other')


# Split the data into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create a pipeline with CountVectorizer and Multinomial Naive Bayes
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train the model and make predictions
model.fit(train_texts, train_labels)
predicted_labels = model.predict(test_texts)

# pickle the model
with open('Naive Bayes.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
print("Model saved successfully.")

# Print predictions for 10 random data points, including two-word cases
print("\nPredictions for 10 Random Data Points:")
random_indices = np.random.choice(len(test_texts), 10, replace=False)
correct_predictions = 0

for index in random_indices:
    input_text = test_texts[index]
    true_label = test_labels[index]
    predicted_label = model.predict([input_text])[0]
    print(f"Input Text: {input_text.strip()} \nTrue Label: {true_label}\nPredicted Label: {predicted_label}\n")

# Print the classification report
print("Classification Report:")
print(metrics.classification_report(test_labels, predicted_labels))