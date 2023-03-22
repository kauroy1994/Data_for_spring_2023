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

        topics = [] #will store the headings and sub-headings
        docs = {} #store the content under the headings and sub-headings

        topics = [line for line in file_lines if ' = ' in line and ' = = ' not in line]
        
        for topic in topics: #for loop to store all the topic contents
            try:
                topic_index = file_lines.index(topic)
                next_topic_index = file_lines.index(topics[topic_index+1])
                docs[topic] = ''.join(file_lines[topic_index+1:next_topic_index])
            except:
                continue

        return topics, docs

    elif info == 'sub':

        topics = [] #will store the headings and sub-headings
        docs = {} #store the content under the headings and sub-headings

        topics = [line for line in file_lines if ' = ' in line and ' = = ' not in line]

        for topic in topics:
            try:
                topic_index = file_lines.index(topic)
                next_topic_index = file_lines.index(topics[topic_index+1])
            
