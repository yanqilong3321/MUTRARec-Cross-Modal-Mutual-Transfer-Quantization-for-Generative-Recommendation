import json
import os
from collections import Counter

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_top1_ratio(codes):
    if not codes: return 0
    cnt = Counter(codes)
    return cnt.most_common(1)[0][1] / len(codes)

def find_best_category():
    base_dir = "/data/yql/workspace/MUTRARec_v1/data/Instruments"
    item_path = os.path.join(base_dir, "Instruments.item.json")
    ind_path = os.path.join(base_dir, "Instruments.index_lemb_v1.json")
    our_path = os.path.join(base_dir, "Instruments.index_lemb_v4.json")
    
    items = load_json(item_path)
    ind_index = load_json(ind_path)
    our_index = load_json(our_path)
    
    # Get all categories
    all_cats = []
    for iid, data in items.items():
        cats = data.get('categories', '').split(',')
        cats = [c.strip() for c in cats if c.strip() and c != 'Musical Instruments']
        all_cats.extend(cats)
    
    top_cats = [c for c, _ in Counter(all_cats).most_common(30)]
    
    print(f"{'Category':<30} | {'Ind Top1':<10} | {'Our Top1':<10} | {'Diff':<10}")
    print("-" * 70)
    
    for cat in top_cats:
        target_items = [iid for iid, data in items.items() if cat in data.get('categories', '')]
        
        ind_codes = [ind_index[iid][0] for iid in target_items if iid in ind_index]
        our_codes = [our_index[iid][0] for iid in target_items if iid in our_index]
        
        if len(ind_codes) < 50: continue # Skip small categories
        
        r1 = get_top1_ratio(ind_codes)
        r2 = get_top1_ratio(our_codes)
        diff = r2 - r1
        
        print(f"{cat:<30} | {r1:.4f}     | {r2:.4f}     | {diff:+.4f}")

if __name__ == "__main__":
    find_best_category()
