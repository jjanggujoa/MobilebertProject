import requests
import time
import csv

app_id = 578080
max_reviews = 500000
cursor = "*"
all_reviews = []
seen_ids = set()
headers = {
    'User-Agent': 'Mozilla/5.0'
}

while len(all_reviews) < max_reviews:
    url = f"https://store.steampowered.com/appreviews/{app_id}?json=1&language=english &filter=recent&num_per_page=100&cursor={cursor}&purchase_type=all"
    response = requests.get(url, headers=headers)
    data = response.json()

    reviews = data.get("reviews", [])
    if not reviews:
        print("더 이상 리뷰가 없습니다.")
        break

    for r in reviews:
        rec_id = r['recommendationid']
        if rec_id not in seen_ids:
            seen_ids.add(rec_id)
            all_reviews.append(r)

    print(f"{len(all_reviews)}개 수집 완료 (중복 제거됨)")

    cursor = data.get("cursor", "")
    cursor = cursor.replace("+", "%2B")

    time.sleep(1)

csv_filename = "pubg_reviews_all.csv"
fieldnames = ['review', 'voted_up', 'timestamp_created', 'author_steamid', 'recommendationid']

with open(csv_filename, "w", newline='', encoding="utf-8-sig") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for r in all_reviews:
        writer.writerow({
            'review': r['review'],
            'voted_up': r['voted_up'],
            'timestamp_created': r['timestamp_created'],
            'author_steamid': r['author']['steamid'],
            'recommendationid': r['recommendationid']
        })

print(f"완료! 총 {len(all_reviews)}개의 중복 제거된 리뷰가 '{csv_filename}'에 저장되었습니다.")
