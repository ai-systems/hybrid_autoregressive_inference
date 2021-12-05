from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer

class Utils:

    def lemmatize(self, string):

        lemmatizer = WordNetLemmatizer()
        process_string = pos_tag(word_tokenize(string))
        temp = []
        for word, pos in process_string:
            if "VB" in pos:
                temp.append(lemmatizer.lemmatize(word,"v"))
            else:
                temp.append(lemmatizer.lemmatize(word))
        return " ".join(temp)  
    
    def init_explanation_bank_lemmatizer(self):
        lemmatization_file = open("lemmatization-en.txt") 
        self.lemmas = {} 
        #saving lemmas
        for line in lemmatization_file: 
            self.lemmas[line.split("\t")[1].lower().replace("\n","")] = line.split("\t")[0].lower()
        return self.lemmas

    def explanation_bank_lemmatize(self, string:str):
        if self.lemmas == None:
            self.init_explanation_bank_lemmatizer()
        temp = []
        for word in string.split(" "):
            if word.lower() in self.lemmas:
                temp.append(self.lemmas[word.lower()])
            else:
                temp.append(word.lower())
        return " ".join(temp)

    def retrieve_entities(self, string):
        retrieved = []
        for entity in self.entities["entities"]:
            if self.lemmatize(string) == entity:
                retrieved.append(entity)
        return retrieved

    def recognize_entities(self, string):
        entities = []
        temp = []
        for word in word_tokenize(string):
            if not word.lower() in stopwords.words("english"):
                temp.append(word.lower())
        tokenized_string = word_tokenize(" ".join(temp))
        head_index = 0
        word_index = 0
        for word in tokenized_string:
            check_index = len(tokenized_string)
            final_entity = ""
            if word_index > head_index:
                head_index = word_index
            while check_index > head_index:
                if len(wordnet.synsets("_".join(tokenized_string[head_index:check_index]))) > 0:
                    final_entity = self.lemmatize(" ".join(tokenized_string[head_index:check_index]))
                    entities.append(final_entity)
                    break
                check_index -= 1
            head_index = check_index
            word_index += 1
        return entities

    def clean_fact(self, fact_explanation):
        fact = []
        for key in fact_explanation:
            if not "[SKIP]" in key and fact_explanation[key] != None:
                fact.append(str(fact_explanation[key]))
        return " ".join(fact)

    def clean_fact_for_overlaps(self, fact_explanation):
        fact = []
        for key in fact_explanation:
            if "FILL" in key or "SKIP" in key or fact_explanation[key] == None:
                continue
            else:
                fact.append(" ".join(str(fact_explanation[key]).split(";")))
        return " ".join(fact)






