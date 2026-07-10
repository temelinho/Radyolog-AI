from totalsegmentator.map_to_binary import class_map

# Tüm damar isimlerini listele
for k, v in class_map["total"].items():
    if any(x in v.lower() for x in ["vein", "artery", "vessel", "aorta", "portal", "splenic", "hepatic", "mesenteric", "celiac"]):
        print(f"{k}: {v}")