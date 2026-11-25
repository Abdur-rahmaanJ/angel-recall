from angel_recall import (
    MemGovernance,
    MemoryAPI,
    MemOperator,
    create_plaintext,
)


class FakeVault:
    def __init__(self, semantic=None, related=None):
        self.semantic = semantic or []
        self.related = related or {}
        self.queries = []

    def semantic_search(self, query, n_results=5, namespace=None, owner=None):
        self.queries.append(query)
        return list(self.semantic)

    def get_related(self, cube_id, relation=None):
        return self.related.get(cube_id, [])


def test_hybrid_retrieve_keeps_relevance_order_and_expands_graph():
    vault = FakeVault()
    api = MemoryAPI(vault, MemGovernance())
    operator = MemOperator(vault, api)

    best = create_plaintext("User lives in Mauritius", owner="u")
    second = create_plaintext("User likes tea", owner="u")
    linked = create_plaintext("User's timezone is UTC+4", owner="u")
    api.query = lambda *args, **kwargs: [best, second]
    vault.related[best.id] = [linked]

    results = operator.hybrid_retrieve("where do I live?", "u", n_results=5)

    assert results == [best, second, linked]


def test_query_falls_back_to_llm_not_word_matching():
    vault = FakeVault()
    api = MemoryAPI(vault, MemGovernance())
    secret = create_plaintext("The secret code is Blue-Raven-42", owner="u")
    api.llm = lambda query, user, namespace, n: [secret]

    assert api.query("what is the code?", "u") == [secret]


def test_query_without_llm_returns_no_guessed_matches():
    vault = FakeVault()
    api = MemoryAPI(vault, MemGovernance())
    create_plaintext("The secret code is Blue-Raven-42", owner="u")

    # No embeddings and no LLM: an empty result is correct, keyword matching is not.
    assert api.query("secret code", "u") == []


def test_scheduler_candidates_are_semantic_not_keyword():
    vault = FakeVault()
    from angel_recall import MemScheduler

    cube = create_plaintext("Budget review on Friday", owner="u")
    vault.semantic = [cube]
    scheduler = MemScheduler(vault)

    assert scheduler._get_plaintext_candidates("when is the budget review?", "u") == [cube]
    assert vault.queries == ["when is the budget review?"]
