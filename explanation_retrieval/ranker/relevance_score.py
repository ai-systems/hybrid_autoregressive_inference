
class RelevanceScore():

    def __init__(self, ranker):
        self.ranker = ranker

    def compute(self, query:str, limit:int):
        relevant_facts = self.ranker.query([query], limit)
        relevance_score = {}
        for i in range(len(relevant_facts)):
            relevance_score[relevant_facts[i]["id"]] = relevant_facts[i]["score"]
        return relevance_score