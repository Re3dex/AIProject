import numpy as np
import os
import glob
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn import metrics
import pickle
from CategoriesGenerator import CategoryBuilder

#Set seed
np.random.seed(42)

# Set the path to your directory containing text files
directory_path1 = "CelebV-Text/face40_details/face40_details_new"
directory_path2 = "FFHQ-Text/00_Female"
directory_path3 = "GeneratedDataset"

text_files_path1 = glob.glob(os.path.join(directory_path1, "*.txt"))
text_files_path2 = glob.glob(os.path.join(directory_path2, "*.txt"))
text_files_path3 = glob.glob(os.path.join(directory_path3, "*.txt"))

all_text_files = text_files_path1 + text_files_path2 + text_files_path3

categories = { #CelebA Annotations in comments
    '5_o_Clock_Shadow': ['stubble', '5 o clock shadow', '5 oh clock shadow'], #5_o_Clock_Shadow
    'Arched_Eyebrows': ['Arched Eyebrows'], #Arched_Eyebrows
    'Attractive': ['beautiful'] , #Attractive
    'Bags_Under_Eyes': ['eye bags', 'bags under eyes'], #Bags_Under_Eyes
    'Bald': ['bald', 'no hair','hairless','hair less'], #Bald
    'Bangs': [ 'short hair', 'crew cut', 'bob cut'], #Bangs
    'Big_Lips': ['big lips', 'large lips'], #Big_Lips
    'Big_Nose': ['bulbous nose', 'big nose', 'large nose', 'huge nose', 'big nose'], #Big_Nose
    'Black_Hair': ['black hair', 'dark hair','black'], #Black_Hair
    'Blonde_Hair': ['blonde','blond','blonde hair','blond hair'], #Blond_Hair
    'Blurry': ['hazy', 'blurry'], #Blurry
    'Brown_Hair': ['brown hair'], #Brown_Hair
    'Bushy_Eyebrows': ['bushy eyebrows', 'big eyebrows', 'thick eyebrows'], #Bushy_Eyebrows
    'Chubby': ['fat'], #Chubby
    'Double_Chin': ['double chin'], #Double_Chin
    'Eyeglasses': ['glasses', 'spectacles'], #Eyeglasses
    'Goatee': ['goatee'], #Goatee
    'Gray_Hair': ['gray hair','grey hair'], #Gray_Hair
    'Heavy_Makeup': ['make up'], #Heavy_Makeup
    'High_Cheekbones': ['high cheekbones'], #High_Cheekbones
    'Male': ['male', 'he', 'man', 'his'], #Male
    'Mouth_Slightly_Open': ['open mouth'], #Mouth_Slightly_Open
    'Mustache': ['mustache','moustache'], #Mustache
    'Narrow_Eyes': ['narrow eyes'], #Narrow_Eyes
    'No_Beard': ['no beard', 'clean shaven','beardless'], #No_Beard
    'Oval_Face': ['oval face'], #Oval_Face
    'Pale_Skin': ['pale', 'pale skin', 'white'], #Pale_Skin
    'Pointy_Nose': ['pointy nose'], #Pointy_Nose
    'Receding_Hairline': ['receding hairline'], #Receding_Hairline
    'Rosy_Cheeks': ['blush'], #Rosy_Cheeks
    'Sideburns': ['sideburns'], #Sideburns
    'Smiling': ['smiling'], #Smiling
    'Straight_Hair': ['straight hair'], #Straight_Hair
    'Wavy_Hair': ['curly hair','wavy'], #Wavy_Hair
    'Wearing_Earrings': ['ear rings', 'earrings'], #Wearing_Earrings
    'Wearing_Hat': ['hat', 'cap'], #Wearing_Hat
    'Wearing_Lipstick': ['lipstick'], #Wearing_Lipstick
    'Wearing_Necklace': ['necklace'], #Wearing_Necklace
    'Wearing_Necktie': ['necktie', 'neck tie'], #Wearing_Necktie
    'Young': ['young', 'youthful', 'teenager'] #Young
}

#Generate More synonyms using Categories Generator Module
# Create an instance of CategoryBuilder
category_builder = CategoryBuilder(categories)

# Build categories with synonyms
category_builder.build_categories_with_synonyms()

# Print the updated categories
CelebA_Categories = category_builder.categories

texts = []
labels = []


files = 0
# Loop through the files and read their contents and assign gold labels
for file_path in all_text_files:
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            words = line.split()
            # Remove punctuation and newline characters, and convert to lowercase
            words = [word.translate(str.maketrans("", "", string.punctuation)).lower() for word in words]
            words = [word.strip() for word in words if word]  # Remove empty strings
            for i in range(len(words)): 
                label_found = False

                # Single word
                for category, keywords in CelebA_Categories.items():
                    if any(keyword == words[i] for keyword in keywords):
                        texts.append(words[i])
                        labels.append(category)
                        label_found = True
                        break

                # Two words
                if i < len(words) - 1 and not label_found:
                    two_words = ' '.join(words[i:i + 2])
                    for category, keywords in CelebA_Categories.items():
                        if any(keyword == two_words for keyword in keywords):
                            texts.append(two_words)
                            labels.append(category)
                            label_found = True
                            break  

                # Three words
                if i < len(words) - 2 and not label_found:
                    three_words = ' '.join(words[i:i + 3])
                    for category, keywords in CelebA_Categories.items():
                        if any(keyword == three_words for keyword in keywords):
                            texts.append(three_words)
                            labels.append(category)
                            label_found = True
                            break  

                # If no label is found, assign "Other"
                if not label_found:
                    texts.append(words[i])
                    labels.append('Other')
    files = files + 1
    if files % 2500 == 0:
        print(f"Progress: {'{0:.2f}'.format((files/len(all_text_files)*100))} / 100")

print("Number of total files read:",files)
print("Training model (15-30m), no progress bar sorry!")

# train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create a pipeline with TfidfVectorizer and RandomForestClassifier
model = make_pipeline(TfidfVectorizer(), RandomForestClassifier(n_estimators=100, random_state=42))

# Train the model and make predictions
model.fit(train_texts, train_labels)
predicted_labels = model.predict(test_texts)

# pickle the model
with open('Forest++.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
print("Model saved successfully.")

# Print predictions for 10 random data points
print("\nPredictions for 10 Random Data Points:")
random_indices = np.random.choice(len(test_texts), 10, replace=False)
correct_predictions = 0
for index in random_indices:
    input_text = test_texts[index]
    true_label = test_labels[index]
    predicted_label = model.predict([input_text])[0]
    print(f"Input Text: {input_text.strip()} \nTrue Label: {true_label}\nPredicted Label: {predicted_label}\n")

# Final Report
print("Classification Report:")
print(metrics.classification_report(test_labels, predicted_labels))
