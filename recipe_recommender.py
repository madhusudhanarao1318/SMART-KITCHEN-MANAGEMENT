
# recipe_recommender.py
RECIPE_DB = [
    {"name": "Fruit Salad","ingredients": {"apple":1, "banana":1}},
    {"name": "Scrambled Eggs & Bottle Sauce","ingredients": {"egg":2, "bottle":1}},
    {"name": "Banana Smoothie","ingredients": {"banana":2, "bottle":1}}
]

def recommend_recipes(inventory_summary: dict, top_k=5):
    candidates = []
    for recipe in RECIPE_DB:
        reqs = recipe["ingredients"]
        satisfied = 0
        total = len(reqs)
        missing = {}
        for ing, cnt in reqs.items():
            have = inventory_summary.get(ing, 0)
            if have >= cnt:
                satisfied += 1
            else:
                missing[ing] = max(0, cnt - have)
        score = satisfied / total
        candidates.append((score, missing, recipe["name"], reqs))

    candidates.sort(key=lambda x: (x[0], -len(x[1])), reverse=True)
    results = []
    for score, missing, name, reqs in candidates[:top_k]:
        if score == 1.0:
            results.append(f"{name} (ready)")
        else:
            missing_str = ", ".join([f"{k} x{v}" for k,v in missing.items()])
            results.append(f"{name} (missing: {missing_str})")
    return results

if __name__ == "__main__":
    inventory = {"apple":2, "banana":1, "egg":3, "bottle":1}
    print(recommend_recipes(inventory))
