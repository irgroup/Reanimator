from functions.utils import get_topic_info

def get_rag_answer(topics_xml, topic_number, rag_bot):
    # Convert topic number to string for matching
    topic_number = str(topic_number)
    
    # Get topic details (query, question, narrative) using get_topic_info
    try:
        query, question, narrative = get_topic_info(topics_xml, topic_number)
    except ValueError as e:
        return {"error": str(e)}
    
    full_question = query + " " + question
    
    # Use RAG bot to get the answer
    rag_response = rag_bot.get_answer(full_question)
    answer = rag_response.get('answer', [])
    contexts = rag_response.get('contexts', [])
    
    # In case of more than 1 generated answers, append them correctly
    if rag_bot._num_answers == 1:
        answers_dict = {"Answer": answer if isinstance(answer, str) else answer[0]}
    else:
        answers_dict = {
            f'Answer No{i+1}': answer[i] if i < len(answer) else None
            for i in range(rag_bot._num_answers)
        }
    
    # Return the aggregated result
    return {
        'Topic Number': topic_number,
        'Query': query,
        'Question': full_question,
        'Narrative': narrative,
        **answers_dict,
        'Contexts': contexts,
    }