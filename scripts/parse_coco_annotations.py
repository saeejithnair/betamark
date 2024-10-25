import json
import os

import tqdm

TRAIN_FILEPATH = os.getenv("TRAIN_FILEPATH")
data = json.load(open(TRAIN_FILEPATH, "r"))
print(data.keys())

# for i in tqdm.trange(1000):

print(data["annotations"][0])

bike_positive_imgs = list()
bike_negative_imgs = list()

for i in tqdm.trange(8000):
    if data["annotations"][i]["category_id"] == 2:
        bike_positive_imgs.append(data["annotations"][i]["image_id"])
    else:
        bike_negative_imgs.append(data["annotations"][i]["image_id"])

bike_positive_imgs = list(set(bike_positive_imgs))
bike_negative_imgs = list(set(bike_negative_imgs))

print("Bike positive images:")
bike_positive_img_len = 0
for i in tqdm.trange(8000):
    if data["images"][i]["id"] in bike_positive_imgs:
        print(data["images"][i]["coco_url"])
        bike_positive_img_len += 1

print("Bike negative images:")

neg_count = 0
for i in tqdm.trange(8000):
    if data["images"][i]["id"] not in bike_positive_imgs:
        print(data["images"][i]["coco_url"])
        neg_count += 1
        if neg_count > bike_positive_img_len:
            break
