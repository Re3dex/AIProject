import speech_recognition as sr
import pickle
from Recorder import AudioRecorder
import os
import glob

# 1. Load the trained model
with open('Forest++.pkl', 'rb') as model_file:
    model1 = pickle.load(model_file)

# 2. Load audio files
# Pre-Recorded Test Files
file_path = "Recordings"
input_audio_file = glob.glob(os.path.join(file_path, "*.wav"))

# Record a test audio audio (uncomment the code below)
# recorder = AudioRecorder()
# recorder.test_record()
# input_audio_file = ["recording.wav"]

# 3. Define the feature vector
feature_vector = {
    '5_o_Clock_Shadow': -1,
    'Arched_Eyebrows': -1,
    'Attractive': -1,
    'Bags_Under_Eyes': -1,
    'Bald': -1,
    'Bangs': -1,
    'Big_Lips': -1,
    'Big_Nose': -1,
    'Black_Hair': -1,
    'Blonde_Hair': -1,
    'Blurry': -1,
    'Brown_Hair': -1,
    'Bushy_Eyebrows': -1,
    'Chubby': -1,
    'Double_Chin': -1,
    'Eyeglasses': -1,
    'Goatee': -1,
    'Gray_Hair': -1,
    'Heavy_Makeup': -1,
    'High_Cheekbones': -1,
    'Male': -1,
    'Mouth_Slightly_Open': -1,
    'Mustache': -1,
    'Narrow_Eyes': -1,
    'No_Beard': -1,
    'Oval_Face': -1,
    'Pale_Skin': -1,
    'Pointy_Nose': -1,
    'Receding_Hairline': -1,
    'Rosy_Cheeks': -1,
    'Sideburns': -1,
    'Smiling': -1,
    'Straight_Hair': -1,
    'Wavy_Hair': -1,
    'Wearing_Earrings': -1,
    'Wearing_Hat': -1,
    'Wearing_Lipstick': -1,
    'Wearing_Necklace': -1,
    'Wearing_Necktie': -1,
    'Young': -1
}

def predict(path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(path) as source:
        # Adjust for ambient noise and listen to the audio
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.record(source)
        try:
            # Use the recognizer to convert audio to text using Google's API
            text = recognizer.recognize_google(audio)
            print("-"*100)
            print("Text from Google API: {}\n".format(text))

            # Convert to a list of words and lowercase
            words = [word.lower() for word in text.split()]
            two_words = [word.lower() for word in text.split()]

            # Generate all permutations of two words
            two_words = [' '.join(pair) for pair in zip(words, words[1:])]

            # Predict classification of words using the trained model
            word_predictions = model1.predict(words)
            phrase_predictions = model1.predict(two_words)

            # Print predictions for each word
            already_predicted = []
            for word, prediction in zip(words, word_predictions):
                if prediction != 'Other' and prediction not in already_predicted:
                    already_predicted.append(prediction)
                    print(f"Word: {word}, Predicted Label: {prediction}")

            for two_word, prediction in zip(two_words, phrase_predictions):
                if prediction != 'Other' and prediction not in already_predicted:
                    already_predicted.append(prediction)
                    print(f"Phrase: {two_word}, Predicted Label: {prediction}")

            return (word_predictions, phrase_predictions, already_predicted)
        
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            # Fallback to Vosk API
            try:
                print("\nGoogle API failed, falling back to vosk!\n")
                text = recognizer.recognize_vosk(audio)
                text = text[14:-3]
                print("Text from Vosk API: {}\n".format(text))

                # Convert to list of words and lowercase
                words = [word.lower() for word in text.split()]
                two_words = [word.lower() for word in text.split()]

                # Generate all permutations of two words
                two_words = [' '.join(pair) for pair in zip(words, words[1:])]

                # Predict classification of words using the trained model
                word_predictions = model1.predict(words)
                phrase_predictions = model1.predict(two_words)

                # Predict classification of two-word categories using nltk because phrases exist (For "Black Hair" - i.e He has black hair, His hair was black)

                # Print predictions for each word
                already_predicted = []
                for word, prediction in zip(words, word_predictions):
                    if prediction != 'Other' and prediction not in already_predicted:
                        already_predicted.append(prediction)
                        print(f"Word: {word}, Predicted Label: {prediction}")

                for two_word, prediction in zip(two_words, phrase_predictions):
                    if prediction != 'Other' and prediction not in already_predicted:
                        already_predicted.append(prediction)
                        print(f"Phrase: {two_word}, Predicted Label: {prediction}")

                return (word_predictions, phrase_predictions, already_predicted)
            
            except sr.UnknownValueError:
                print("Vosk Speech Recognition could not understand audio")
                return None
            except sr.RequestError as e:
                print("Could not request results from Vosk Speech Recognition service; {0}".format(e))
                return None
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return None
        
def prediction_results(paths = input_audio_file, feature_vector = feature_vector):
    for path in paths:
        instance_vector = feature_vector

        predictions = predict(path)
        word_predictions = predictions[0]
        phrase_predictions = predictions[1]
        already_predicted = predictions[2]

        for prediction in word_predictions:
            if prediction != 'Other' and instance_vector[prediction] == -1 :
                instance_vector[prediction] = 1

        for prediction in phrase_predictions:
            if prediction != 'Other' and instance_vector[prediction] == -1 :
                instance_vector[prediction] = 1

        print("\nNo. of Features extracted:",len(already_predicted))
        print("\n Feature_vector:\n",instance_vector)

if __name__ == '__main__':
    prediction_results()