"""
Text processing utilities.
"""

def clean_text(text):
    """
    Clean and normalize text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    return " ".join(text.lower().split())

def tokenize(text):
    """
    Tokenize text into words.
    
    Args:
        text (str): Input text
        
    Returns:
        list: List of words
    """
    return clean_text(text).split()
