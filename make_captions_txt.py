# make_captions_txt.py

import json

def extract_captions(annotations_file, output_txt):
    with open(annotations_file, "r") as f:
        data = json.load(f)

    captions = [ann['caption'].strip() for ann in data['annotations']]

    print(f"Extracting {len(captions)} captions to {output_txt}")

    with open(output_txt, "w", encoding="utf-8") as out_f:
        for caption in captions:
            out_f.write(caption + "\n")

if __name__ == "__main__":
    extract_captions("./data/annotations/captions_train2017.json", "captions.txt")

