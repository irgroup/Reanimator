class CombinedRetriever:
    def __init__(self, retriever_text, retriever_table, total_k=10, text_ratio=0.5):
        self.retriever_text = retriever_text
        self.retriever_table = retriever_table
        self.total_k = total_k
        self.text_ratio = text_ratio

    def invoke(self, question):
        # Calculate how many docs to get from each source
        num_text = int(round(self.total_k * self.text_ratio))
        num_table = self.total_k - num_text

        # Retrieve documents from both retrievers
        docs_text = self.retriever_text.invoke(question)[:num_text]
        docs_table = self.retriever_table.invoke(question)[:num_table]

        # Combine the results
        interleaved_docs = []
        while docs_text or docs_table:
            if docs_text:
                interleaved_docs.append(docs_text.pop(0))
            if docs_table:
                interleaved_docs.append(docs_table.pop(0))
        return interleaved_docs