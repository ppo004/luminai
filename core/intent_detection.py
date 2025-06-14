"""
Intent detection module for query classification.
"""

def detect_intent(query_text):

    query_lower = " ".join(query_text.lower().split())
    words = query_lower.split()
    
    intent_keywords = {
        "summarization": {
            "summarize": 4, "summary": 4, "overview": 3, "brief": 3, "give me a": 2, "short": 2, "recap": 2,
            "status": 2, "progress": 2, "update": 2, "meeting": 1, "project": 1
        },
        "explanation": {
            "explain": 4, "how": 4, "why": 4, "what is": 3, "break down": 3, "describe": 2, "tell me": 2, "clarify": 2,
            "works": 3, "process": 3, "deploy": 2, "authentication": 2, "jwt": 2, "database": 2, "api": 2, 
            "microservices": 2, "integration": 1, "setup": 1, "function": 1
        },
        "list": {
            "list": 4, "steps": 4, "items": 3, "details": 3, "what are": 3, "show": 2, "all": 2, "outline": 2,
            "tasks": 3, "action items": 3, "services": 2, "endpoints": 2, "components": 2, "tools": 2, 
            "requirements": 2, "features": 1, "schema": 1, "collections": 1
        }
    }
    
    scores = {"summarization": 0, "explanation": 0, "list": 0}
    for intent, keywords in intent_keywords.items():
        for keyword, weight in keywords.items():
            if " " in keyword:
                if keyword in query_lower:
                    scores[intent] += weight
            else:
                if keyword in words:
                    scores[intent] += weight
    
    print("The intent scores", scores)
    top_intent = max(scores, key=scores.get)
    max_score = scores[top_intent]
    
    if max_score < 3:
        return "general"
    
    sorted_scores = sorted(scores.values(), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[0] == sorted_scores[1]:
        return "general"
    
    return top_intent

def get_instruction_and_format(intent):
    if intent == "summarization":
        instruction = "Summarize in 6-7 sentences."
        format_instruction = "Use plain text."
    elif intent == "explanation":
        instruction = "Explain step-by-step."
        format_instruction = "Use numbered steps if applicable."
    elif intent == "list":
        instruction = "Provide a detailed response."
        format_instruction = "Use bullet points for key items or steps."
    else: 
        instruction = "Answer directly."
        format_instruction = "Use plain text; if the query is unclear, ask for clarification."
    
    return instruction, format_instruction
