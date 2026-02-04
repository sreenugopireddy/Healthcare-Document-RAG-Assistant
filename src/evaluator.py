import json

def load_eval_queries():
    with open("evaluation_queries.json") as f:
        return json.load(f)

def run_eval(agent):
    queries = load_eval_queries()
    results = []

    for q in queries:
        ans = agent.run(q)
        results.append({
            "query": q,
            "answer": ans
        })

    return results
