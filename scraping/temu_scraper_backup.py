from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import csv
import time
import os
import re

options = Options()
options.add_argument("--start-maximized")

driver = webdriver.Chrome(options=options)

url = "https://www.trustpilot.com/review/temu.com"
driver.get(url)

print("Waiting for Trustpilot...")
time.sleep(10)

print("Title:", driver.title)

reviews = driver.find_elements(By.CSS_SELECTOR, "article")

print("Reviews found:", len(reviews))

data = []

for i, review in enumerate(reviews, start=1):

    try:
        # -------------------------
        # Reviewer
        # -------------------------
        reviewer_elements = review.find_elements(
            By.CSS_SELECTOR,
            "[data-consumer-name-typography='true']"
        )

        reviewer = (
            reviewer_elements[0].text.strip()
            if reviewer_elements
            else ""
        )

        # -------------------------
        # Date
        # -------------------------
        date_elements = review.find_elements(
            By.CSS_SELECTOR,
            "time"
        )

        date = ""

        if date_elements:
            date = date_elements[0].get_attribute("datetime")

            if not date:
                date = date_elements[0].text.strip()

        # -------------------------
        # Rating
        # -------------------------
        rating = ""

        rating_elements = review.find_elements(
            By.CSS_SELECTOR,
            "img[alt*='Rated']"
        )

        if rating_elements:

            rating_text = rating_elements[0].get_attribute("alt")

            match = re.search(
                r"Rated (\d+) out of 5",
                rating_text
            )

            if match:
                rating = match.group(1)

        # -------------------------
        # Review Text
        # -------------------------
        text_elements = review.find_elements(
            By.CSS_SELECTOR,
            "[data-relevant-review-text-typography='true']"
        )

        review_text = ""

        if text_elements:
            review_text = text_elements[0].text.strip()

        # -------------------------
        # Save review
        # -------------------------
        if reviewer or review_text:

            data.append({
                "company": "Temu",
                "reviewer": reviewer,
                "rating": rating,
                "date": date,
                "review_text": review_text
            })

            print(
                f"Review {i}: "
                f"{reviewer} | "
                f"Rating: {rating}"
            )

    except Exception as e:

        print(f"Review {i} error:", e)


# -------------------------
# Save CSV
# -------------------------

os.makedirs("data", exist_ok=True)

file_path = "data/temu_reviews.csv"

with open(
    file_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "company",
            "reviewer",
            "rating",
            "date",
            "review_text"
        ]
    )

    writer.writeheader()
    writer.writerows(data)


print()
print("==============================")
print("SCRAPING COMPLETED")
print("==============================")
print("Reviews found:", len(reviews))
print("Reviews saved:", len(data))
print("File:", file_path)

input("Press Enter to close browser...")

driver.quit()