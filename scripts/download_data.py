"""Download the Olist Brazilian E-commerce dataset (CC BY-NC-SA 4.0) into data/raw/.

Source: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
Mirror used here (no Kaggle login needed): Hugging Face dataset aviahYadler/Olist_Ecommerce_Dataset
"""

from pathlib import Path

import requests

MIRROR = "https://huggingface.co/datasets/aviahYadler/Olist_Ecommerce_Dataset/resolve/main"
FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def download(force: bool = False) -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in FILES:
        dest = RAW_DIR / name
        if dest.exists() and not force:
            print(f"skip  {name} (exists)")
        else:
            print(f"fetch {name}")
            with requests.get(f"{MIRROR}/{name}", stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                tmp.rename(dest)
        paths.append(dest)
    return paths


if __name__ == "__main__":
    download()
