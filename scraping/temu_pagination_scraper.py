from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import csv
import time
import os
import re


# =========================
# SETTINGS
# =========================

COMPANY = "Temu"
INDUSTRY = "Online Shopping"

BASE_URL = "https://www.trustpilot.com/review/temu.com"

TARGET_REVIEWS = 500


# =========================
# BROWSER
# =========================

options = Options()
options.add_argument("--start-maximized")

driver = webdriver.Chrome(options=options)


# =========================
# DATA
# =========================

data = []
seen_reviews = set()

page = 1


# =========================
# PAGINATION
# =========================

while len(data) < TARGET_REVIEWS:

    if page == 1:
        url = BASE_URL
    else:
        url = BASE_URL + f"?page={page}"

    print()
    print("==============================")
    print("Opening page:", page)
    print("URL:", url)
    print("==============================")

    driver.get(url)

    print("Waiting for Trustpilot...")
    time.sleep(8)

    print("Title:", driver.title)

    # Stop if Trustpilot sends us to login
    if "Trustpilot Login" in driver.title:
        print("Trustpilot login page detected.")
        print("Stopping scraper.")
        break

    reviews = driver.find_elements(
        By.CSS_SELECTOR,
        "article"
    )

    print("Reviews found on page:", len(reviews))

    if not reviews:
        print("No reviews found. Stopping.")
        break


    page_count = 0


    # =========================
    # EXTRACT REVIEWS
    # =========================

    for review in reviews:

        if len(data) >= TARGET_REVIEWS:
            break

        try:

            # -------------------------
            # REVIEW TEXT
            # -------------------------

            text_elements = review.find_elements(
                By.CSS_SELECTOR,
                "[data-relevant-review-text-typography='true']"
            )

            review_text = ""

            if text_elements:
                review_text = text_elements[0].text.strip()

            # Fallback
            if not review_text:

                paragraphs = review.find_elements(
                    By.TAG_NAME,
                    "p"
                )

                for p in paragraphs:

                    text = p.text.strip()

                    if text and len(text) > 20:
                        review_text = text
                        break


            if not review_text:
                continue


            # -------------------------
            # REVIEW TITLE
            # -------------------------

            title = ""

            title_elements = review.find_elements(
                By.CSS_SELECTOR,
                "[data-review-title-typography='true']"
            )

            if title_elements:
                title = title_elements[0].text.strip()


            # -------------------------
            # RATING
            # -------------------------

            rating = ""

            rating_elements = review.find_elements(
                By.CSS_SELECTOR,
                "img[alt*='Rated']"
            )

            if rating_elements:

                rating_text = rating_elements[0].get_attribute(
                    "alt"
                )

                match = re.search(
                    r"Rated (\d+) out of 5",
                    rating_text
                )

                if match:
                    rating = match.group(1)


            # -------------------------
            # DATE
            # -------------------------

            date = ""

            date_elements = review.find_elements(
                By.CSS_SELECTOR,
                "time"
            )

            if date_elements:

                date = date_elements[0].get_attribute(
                    "datetime"
                )

                if not date:
                    date = date_elements[0].text.strip()


            # -------------------------
            # DUPLICATE CHECK
            # -------------------------

            review_key = (
                COMPANY,
                review_text,
                rating,
                date
            )

            if review_key in seen_reviews:
                continue

            seen_reviews.add(review_key)


            # -------------------------
            # SAVE
            # -------------------------

            data.append({
                "Company": COMPANY,
                "Industry": INDUSTRY,
                "Review Title": title,
                "Review Text": review_text,
                "Rating": rating,
                "Date": date
            })

            page_count += 1

        except Exception as e:

            print(
                "Error extracting review:",
                e
            )


    print(
        f"New reviews collected from page {page}: {page_count}"
    )

    print(
        f"Total reviews collected: "
        f"{len(data)}/{TARGET_REVIEWS}"
    )

    page += 1


# =========================
# SAVE CSV
# =========================

os.makedirs("data", exist_ok=True)

file_path = "data/temu_raw_reviews.csv"


with open(
    file_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "Company",
            "Industry",
            "Review Title",
            "Review Text",
            "Rating",
            "Date"
        ]
    )

    writer.writeheader()
    writer.writerows(data)


# =========================
# RESULT
# =========================

print()
print("==============================")
print("SCRAPING COMPLETED")
print("==============================")
print("Company:", COMPANY)
print("Industry:", INDUSTRY)
print("Total reviews:", len(data))
print("Pages processed:", page - 1)
print("File:", file_path)

input("Press Enter to close browser...")

driver.quit()