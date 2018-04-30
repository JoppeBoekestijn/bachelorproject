# Joppe Boekestijn, s2754215
import requests
import sys

url = 'https://query.wikidata.org/sparql'

questionNumber = int(sys.argv[1])


def request_and_print(sparql_query):
    """
    :param sparql_query: query to sparql
    print answer
    """
    data = requests.get(url,
                        params={'query': sparql_query, 'format': 'json'}).json()

    for item in data['results']['bindings']:
        for var in item:
            print('{}'.format(item[var]['value']))


if questionNumber == 1:
    print("What is the highest point in Africa?")
    query = '''
    SELECT ?itemLabel WHERE {
    wd:Q15 wdt:P610 ?item .
    
    SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en" .
    }
    }'''
    request_and_print(sparql_query=query)
elif questionNumber == 2:
    print("What are three countries that contain the Himalayas?")
    query = '''
    SELECT ?itemLabel WHERE {
    wd:Q5451 wdt:P17 ?item .
    
    SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en" .
    }
    }
    LIMIT 3'''
    request_and_print(sparql_query=query)
elif questionNumber == 3:
    print("What is the deepest point in the People's republic of China?")
    query = '''
    SELECT ?itemLabel WHERE {
    wd:Q148 wdt:P1589 ?item .

    SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en" .
    }
    }'''
    request_and_print(sparql_query=query)
elif questionNumber == 4:
    print("In what country is the Grand Canyon situated?")
    query = '''
    SELECT ?itemLabel WHERE {
    wd:Q118841 wdt:P17 ?item .

    SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en" .
    }
    }'''
    request_and_print(sparql_query=query)
elif questionNumber == 5:
    print("What is the sister city of Leeuwarden?")
    query = '''
    SELECT ?itemLabel WHERE {
    wd:Q25390 wdt:P190 ?item .

    SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en" .
    }
    }'''
    request_and_print(sparql_query=query)
elif questionNumber == 6:
    print("What is the capital of the Netherlands?")
    query = '''
    SELECT ?itemLabel WHERE {
    wd:Q55 wdt:P36 ?item .

    SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en" .
    }
    }'''
    request_and_print(sparql_query=query)
elif questionNumber == 7:
    print("What is the Atlantic ocean named after?")
    query = '''
    SELECT ?itemLabel WHERE {
    wd:Q97 wdt:P138 ?item .

    SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en" .
    }
    }'''
    request_and_print(sparql_query=query)
elif questionNumber == 8:
    print("In what country was Carthage situated?")
    query = '''
    SELECT ?itemLabel WHERE {
    wd:Q6343 wdt:P17 ?item .

    SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en" .
    }
    }'''
    request_and_print(sparql_query=query)
elif questionNumber == 9:
    print("What are the official languages of Ireland?")
    query = '''
    SELECT ?itemLabel WHERE {
    wd:Q27 wdt:P37 ?item .

    SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en" .
    }
    }'''
    request_and_print(sparql_query=query)
elif questionNumber == 10:
    print("What is the country of origin of pasta?")
    query = '''
    SELECT ?itemLabel WHERE {
    wd:Q178 wdt:P495 ?item .

    SERVICE wikibase:label {
        bd:serviceParam wikibase:language "en" .
    }
    }'''
    request_and_print(sparql_query=query)
else:
    print('Please enter value between 1 and 10')



