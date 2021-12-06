
class ExplanatoryPower():

    def __init__(self, ranker, explanations_corpus):
        self.ranker = ranker
        self.EKB = explanations_corpus

    def compute(self, q_id:str, query:str, sim_questions_limit:int, facts_limit:int):
        similar_questions = self.ranker.question_similarity([query])[:sim_questions_limit]
        explanatory_power = {}
        for i in range(len(similar_questions)):
            if similar_questions[i]["id"] == q_id:
                continue
            for exp in self.EKB[similar_questions[i]["id"]]['_explanation']:
                if not exp in explanatory_power:
                    explanatory_power[exp] = 0
                if similar_questions[i]["score"] > 0:
                	explanatory_power[exp] += similar_questions[i]["score"]
        sorted_explanatory_power = {}
        for key in sorted(explanatory_power, key=explanatory_power.get, reverse=True)[:facts_limit]:
            sorted_explanatory_power[key] = explanatory_power[key]
        return sorted_explanatory_power