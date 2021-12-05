
class Utils:

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

    def clean_fact(self, fact_explanation):
        fact = []
        for key in fact_explanation:
            if not "[SKIP]" in key and fact_explanation[key] != None:
                fact.append(str(fact_explanation[key]))
        return " ".join(fact)





