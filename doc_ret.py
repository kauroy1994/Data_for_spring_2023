from wiki_exps import get_info, get_all_subtopics_docs, get_topic_docs
from autocomplete import ffn_complete

topics_of_interest = [' = Valkyria Chronicles III = ', ' = Tower Building of the Little Rock Arsenal = ']

docs = get_info(info='main')
main_texts = get_topic_docs(docs,topics_of_interest[0])

sub_docs = get_info(info='sub')
sub_texts = get_all_subtopics_docs(sub_docs,topics_of_interest[0])
