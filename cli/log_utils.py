def log_results(title: str, results: list[dict]):
    print(f"{title}:")
    for res in results:
        print(f"    * {res["document"]["title"]}:")
        safe_log_int("keyword_rank", res)
        safe_log_int("semantic_rank", res)
        safe_log_float("rrf_score", res)
        safe_log_float("re_rank", res)
        
def safe_log_int(key: str, res: dict):
    value : str = res[key] if key in res else "-"
    print(f"        * {key}: {value}")
    
def safe_log_float(key: str, res: dict):
    value : str = f"{res[key]:.4f}" if key in res else "-"
    print(f"        * {key}: {value}")