from tqdm import tqdm

def get_info(info='main'):

    '''
    extracts headings and sub-headings from the wiki docs

    param-info:
                         takes values from {'main','sub',subsub'}
                         main for main headings
                         sub for sub headings
                         subsub for sub headings within sub headings
    
    '''

    if info not in ['main','sub','subsub']:

        help_string = """param-info:
                         takes values from {'main','sub',subsub'}
                         main for main headings
                         sub for sub headings
                         subsub for sub headings within sub headings
                      """

        print (help_string)
        exit()

    with open('wikitext-2/wiki.train.tokens') as f:

        file_lines = f.read().splitlines()


    if info == 'main':

        docs = {} #store the content under the headings and sub-headings

        topics = [line for line in file_lines if ' = ' in line and ' = = ' not in line]
        
        print ('Extracting documents from the wiki dumps ...')
        for topic in tqdm(topics): #for loop to store all the topic contents
            try:
                topic_index = file_lines.index(topic)
                next_topic_index = file_lines.index(topics[topic_index+1])
                docs[topic] = ''.join(file_lines[topic_index+1:next_topic_index])
            except:
                continue

        return docs

    elif info == 'sub':

        docs = {} #store the content under the headings and sub-headings

        topics = [line for line in file_lines if ' = ' in line and ' = = ' not in line]
        
        print ('Extracting documents and sub documents from the wiki dumps ...')
        for topic in tqdm(topics):
            try:
                topic_index = file_lines.index(topic)
                next_topic_index = file_lines.index(topics[topic_index+1])
                topic_content_lines = file_lines[topic_index+1:next_topic_index]

                subtopics = [line for line in topic_content_lines if ' = = ' in line and ' = = = ' not in line]

                for subtopic in subtopics:
                    try:
                        subtopic_index = file_lines.index(subtopic)
                        next_subtopic_index = file_lines.index(subtopics[subtopic_index+1])
                        docs[(topic,subtopic)] = ''.join(file_lines[subtopic_index+1:next_subtopic_index])
                    except:
                        continue
            except:
                continue
        return docs

    else:

        print ('no executable function available for this script\nExiting program ...')
        exit()

def get_all_subtopics_docs(docs,topic):

    '''
    returns doc relevant to the topic and its subtopics
    param-info:

                docs = document dictionary returned by get_info(info='sub')
                topic = a topic name e.g., ' = Tower Building of the Little Rock Arsenal = '
    '''

    topic_docs = {} #stores document contents
    relevant_keys = [key for key in list(docs.keys()) if key[0] == topic]

    for key in relevant_keys:
        topic_docs[key] = docs[key]

    return topic_docs

def get_topic_docs(docs, topic):

    '''
    returns doc relevant to the topic
    param-info:

                docs = document dictionary returned by get_info(info='main')
                topic = a topic name e.g., ' = Tower Building of the Little Rock Arsenal = '
    '''

    topic_docs = {} #stores document contents
    relevant_keys = [key for key in list(docs.keys()) if key == topic]

    for key in relevant_keys:
        topic_docs[key] = docs[key]

    return topic_docs
