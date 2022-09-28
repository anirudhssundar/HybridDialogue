def error_analysis(turn_id, dataset, num_retrieve=10):
    turn = dataset.get_turn(turn_id)
    query = turn['current_query']
    correct_candidate = candidates[turn['correct_next_cands_ids'][0]]

    correct_source = correct_candidate['page_key'] or correct_candidate['table_key'].rsplit('_', 1)[0]

    correct_source = correct_source.replace("_", ' ').lower()
    # correct_source = correct_source.lower()
    files = list(os.listdir(dataset.orig_data_dir))
    files_prepped = list(map(lambda x: prep_file_names(x), files))
    tokenized_corpus = [doc.split(" ") for doc in files_prepped]
    bm25 = BM25Okapi(tokenized_corpus)

    query = query.lower()
    tokenized_query = query.split(" ")
    results = bm25.get_top_n(tokenized_query, tokenized_corpus, n=num_retrieve)
    results = list(map(lambda x: ' '.join(x), results))

    print(f"query: {query}")
    print(f"correct source: {correct_source}")
    print(f"results:{results}") 



def get_rank(query, correct_source, num_retrieve=10):
    files = list(os.listdir(dataset.orig_data_dir))
    files_prepped = list(map(lambda x: prep_file_names(x), files))
    tokenized_corpus = [doc.split(" ") for doc in files_prepped]
    bm25 = BM25Okapi(tokenized_corpus)

    query = query.lower()
    tokenized_query = query.split(" ")
    results = bm25.get_top_n(tokenized_query, tokenized_corpus, n=num_retrieve)
    results = list(map(lambda x: ' '.join(x), results))

    if correct_source not in results:
        return num_retrieve + 1
    else:
        return results.index(correct_source) + 1


def passage_ranking(query, dataset):
    files = list(os.listdir(dataset.orig_data_dir))
    files_prepped = list(map(lambda x: prep_file_names(x), files))
    tokenized_corpus = [doc.split(" ") for doc in files_prepped]
    bm25 = BM25Okapi(tokenized_corpus)

    query = query.lower()
    tokenized_query = query.split(" ")
    # doc_scores = bm25.get_scores(tokenized_query)
    top_result = ' '.join(bm25.get_top_n(tokenized_query, tokenized_corpus, n=1)[0])
    return top_result


