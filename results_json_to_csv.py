import json
import os


XVIEW_CONFIG = os.environ["XVIEW_CONFIG"]
EPOCHS = int(os.environ["XVIEW_EPOCHS"])

PREFIX = "val_scores/" +XVIEW_CONFIG+ "/epoch_"

print("score", "damage_f1", "localization_f1",
      "damage_f1_no_damage", "damage_f1_minor_damage",
      "damage_f1_major_damage", "damage_f1_destroyed")
for i in range(EPOCHS):
    r = json.load(open(PREFIX + str(i) + ".json", "r"))
    print(r["score"], r["damage_f1"],
          r["localization_f1"],
          r["damage_f1_no_damage"],
          r["damage_f1_minor_damage"],
          r["damage_f1_major_damage"],
          r["damage_f1_destroyed"])
