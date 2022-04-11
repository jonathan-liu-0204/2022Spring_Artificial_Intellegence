from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
import re
import string


def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)

    # Begin your code (Part 0)

    ## get rid of chinese
    text1 = re.sub('[\u4e00-\u9fa5]', '', text)

    ## get rid of unrelated symbols, such as < . * ? > ! $( ) * % @
    symbols = re.compile(r'<.*?>!$()%@')
    text2 = symbols.sub('',text1)

    ## get rid of Upper letters, make all of them into lower case
    text3 = text2.lower()

    ## get rid of all the numbers
    text_final = re.sub(r'[0-9]+', '', text3)

    preprocessed_text = text_final
    
    # End your code

    return preprocessed_text