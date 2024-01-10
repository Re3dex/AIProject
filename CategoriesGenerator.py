import nltk # For finding the synonyms of words
nltk.download('wordnet')
from nltk.corpus import wordnet 

import spacy # For calculating word similarity
spacy_model = spacy.load("en_core_web_md")

import mobypy as moby # For validating word similarity and adding any missed synonyms



class CategoryBuilder:

    def __init__(self, categories, threshold = 0.55):
        self.categories_old = categories
        self.categories = categories
        self.similarity_threshold = threshold # How similar in meaning must the additions be? Between 1 and 0, 1 being most similar and vice versa

    def get_similarity_spacy(self, word1, word2):
        # Create spaCy Doc objects for the two words
        doc1 = spacy_model(word1.lower())
        doc2 = spacy_model(word2.lower())

        # Calculate the similarity between the two words
        similarity = doc1.similarity(doc2)
        return similarity

    def get_similarity_wordnet(self, word1, word2):
        # Get synsets for each word
        synsets_word1 = wordnet.synsets(word1.lower())
        synsets_word2 = wordnet.synsets(word2.lower())

        # Calculate Wu-Palmer similarity between the most common synsets of each word
        if synsets_word1 and synsets_word2:
            similarity = synsets_word1[0].wup_similarity(synsets_word2[0])
            return similarity
        else:
            # Handle the case where one or both words have no synsets
            return 0

    def get_synonyms(self,word):
        synonyms = set()

        #Wordnet
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                similarity1 = self.get_similarity_spacy(word,lemma.name())
                similarity2 = self.get_similarity_wordnet(word,lemma.name())
                print(f"[WN] - {word}, {lemma.name()}[{syn.pos()}] | Similarity: [S]{'{0:.2f}'.format(similarity1)} [W]{'{0:.2f}'.format(similarity2)}")
                if ((similarity1 >= self.similarity_threshold and similarity2 >= self.similarity_threshold) or
                    ((similarity1 >= 0.95 or similarity2 >= 0.95) and (similarity1 > 0 and similarity2 > 0))):    
                    synonyms.add(lemma.name())

        #Moby
        moby_synonyms = moby.synonyms(word)
        for syn in moby_synonyms:
                similarity1 = self.get_similarity_spacy(word,syn)
                similarity2 = self.get_similarity_wordnet(word,syn)
                print(f"[MB] - {word}, {syn} | Similarity: [S]{'{0:.2f}'.format(similarity1)} [W]{'{0:.2f}'.format(similarity2)}")
                if ((similarity1 >= self.similarity_threshold and similarity2 >= self.similarity_threshold) or
                    ((similarity1 >= 0.95 or similarity2 >= 0.95) and (similarity1 > 0 and similarity2 > 0))):                    
                    synonyms.add(syn)

        return list(synonyms)

    def build_categories_with_synonyms(self):
        for category, synonyms in self.categories.items():
            synonym_set = set(synonyms)

            # Tokenize the category and the already provided synonyms
            words = nltk.word_tokenize(category)
            for w in synonym_set:
                w = w.replace(' ','_')
                w_token = nltk.word_tokenize(w)
                words.append(w_token[0])
            
            # Flatten the list of words (category + hard-coded synonyms)
            self.flatten(words)

            # Get and append synonyms for each word in the list
            for word in words:
                synonym_set.update([syn.lower().replace('_', ' ').replace('-', ' ') for syn in self.get_synonyms(word)])

            self.categories[category] = list(synonym_set)

    def print_categories(self):
        for key, value in self.categories.items():
            print("\n", key, value)

    def flatten(self,matrix):
        return [item for row in matrix for item in row]


# 1. Prefer single word category names such as bearlesss instead of no beard
# 2. Works with just the category name(dict key) but adding more synonyms in the list(dict value) will imrpove its generation
# NOTE: Does not work with slangs and "two worded" words i.e arched eyebrows, add those yourself

categories = { #CelebA Annotations in comments
    'Stubble': ['stubble', '5 o clock shadow', '5 oh clock shadow'], #5_o_Clock_Shadow
    'Arched_Eyebrows': ['Arched Eyebrows'], #Arched_Eyebrows
    'Attractive': ['beautiful'] , #Attractive
    'Eyebags': ['eye bags', 'bags under eyes'], #Bags_Under_Eyes
    'Bald': ['bald', 'no hair','hairless','hair less'], #Bald
    'Fringe': [ 'short hair', 'crew cut', 'bob cut'], #Bangs
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
    'Open_Mouth': ['open mouth'], #Mouth_Slightly_Open
    'Mustache': ['mustache','moustache'], #Mustache
    'Narrow_Eyes': ['narrow eyes'], #Narrow_Eyes
    'No_Beard': ['no beard', 'clean shaven','beardless'], #No_Beard
    'Oval_Face': ['oval face'], #Oval_Face
    'Pale_Skin': ['pale', 'pale skin', 'white'], #Pale_Skin
    'Pointy_Nose': ['pointy nose'], #Pointy_Nose
    'Receding_Hairline': ['receding hairline'], #Receding_Hairline
    'Blush': ['rosy cheeks'], #Rosy_Cheeks
    'Sideburns': ['sideburns'], #Sideburns
    'Smiling': ['smiling'], #Smiling
    'Straight_Hair': ['straight hair'], #Straight_Hair
    'Wavy_Hair': ['curly hair'], #Wavy_Hair
    'Earrings': ['ear rings', 'earrings'], #Wearing_Earrings
    'Hat': ['hat', 'cap'], #Wearing_Hat
    'Lipstick': ['lipstick'], #Wearing_Lipstick
    'Necklace': ['necklace'], #Wearing_Necklace
    'Necktie': ['necktie', 'neck tie'], #Wearing_Necktie
    'Young': ['young', 'youthful', 'teenager'] #Young
}

if __name__ == '__main__':
    # Create an instance of CategoryBuilder
    category_builder = CategoryBuilder(categories)

    # Build categories with synonyms
    category_builder.build_categories_with_synonyms()

    # Print the updated categories
    category_builder.print_categories()