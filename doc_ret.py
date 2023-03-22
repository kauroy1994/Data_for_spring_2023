from wiki_exps import get_info, get_all_subtopics_docs, get_topic_docs
from autocomplete import ffn_complete, Tokenizer, Dataloader

topics_of_interest = [' = Valkyria Chronicles III = ', ' = Tower Building of the Little Rock Arsenal = ']

docs = get_info(info='main')
main_texts = get_topic_docs(docs,topics_of_interest[0])

sub_docs = get_info(info='sub')
sub_texts = get_all_subtopics_docs(sub_docs,topics_of_interest[0])

all_text = topics_of_interest[0]+main_texts[topics_of_interest[0]][:100]

chars = list(set(all_text))
tokenizer = Tokenizer(tokens = chars)
data_loader = Dataloader(tokenizer = tokenizer,
                         text = all_text)

n_tokens = tokenizer.n_tokens
emb_size = 100
context_size = len(data_loader.get_batch())
model = ffn_complete(n_tokens = n_tokens,
                     emb_size = emb_size,
                     context_size = context_size)

model.train(data_loader)
