import argparse
import json
import logging
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Dict, List
from urllib.parse import urljoin

from abc_xml_converter import convert_xml2abc
from dotenv import load_dotenv
from pdf2image import convert_from_path
from playwright.sync_api import Page, sync_playwright

LOGIN_URL = "https://www.musescore.com"
SHEET_MUSIC_URL = "https://musescore.com/sheetmusic/free-download"
HEADLESS = True
logger = logging.getLogger(__name__)


@dataclass
class ScraperConfig:
    num_workers: int
    num_scores: int
    output_dir: Path


def parse_args():
    parser = argparse.ArgumentParser(description="scraping script")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of scraper worker threads",
    )
    parser.add_argument(
        "--num-scores", type=int, default=1000, help="Number of scores to scrape"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save scraped data"
    )
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    if args.num_scores < 1:
        parser.error("--num-scores must be at least 1")
    return ScraperConfig(
        num_workers=args.num_workers,
        num_scores=args.num_scores,
        output_dir=Path(args.output_dir),
    )


def get_login_credentials():
    load_dotenv()
    login_email = os.getenv("LOGIN_EMAIL")
    login_password = os.getenv("LOGIN_PASSWORD")
    if not login_email or not login_password:
        raise ValueError("LOGIN_EMAIL and LOGIN_PASSWORD must be set in .env file")
    return login_email, login_password


def login(page: Page, email: str, password: str) -> None:
    page.goto(LOGIN_URL)
    page.wait_for_selector(
        "button[class='TtlUw TtlUw DHBDz HFvdW plVkZ wXNik utani u_VDg']"
    )
    page.click("button[class='TtlUw TtlUw DHBDz HFvdW plVkZ wXNik utani u_VDg']")
    page.wait_for_selector("form[id='user-login-form']")
    page.wait_for_selector("#username")
    page.wait_for_selector("#password")
    page.wait_for_selector("button[type='submit']")
    page.locator("#username").fill(email)
    page.locator("#password").fill(password)
    page.locator("button[class='vs3kE YAo5i Ux6Ko bOSK0 otHSn']").click()

    page.wait_for_load_state("networkidle")


def parse_metadata(meta_str: List[str], link: str) -> Dict[str, object]:
    metadata = {}
    for m in meta_str:
        if "pages" in m:
            metadata["pages"] = int(m.split()[0])
        elif "saves" in m:
            cnt = m.split()[0]
            if cnt.endswith("K"):
                cnt = int(float(cnt[:-1]) * 1000)
            elif cnt.endswith("M"):
                cnt = int(float(cnt[:-1]) * 1000000)
            elif cnt.endswith(tuple(str(i) for i in range(10))):
                cnt = int(cnt)
            else:
                raise ValueError(f"Unexpected saves count format: {cnt}")
            metadata["saves"] = cnt
    broken_link = link.split("/")
    metadata["score_id"] = f"{broken_link[-3]}_{broken_link[-1]}"
    metadata["link"] = link
    return metadata


def head_worker(config: ScraperConfig, queue: Queue, email: str, password: str):
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=HEADLESS)
    page = browser.new_page()
    NEXT_URL = SHEET_MUSIC_URL
    current_page = 1
    n_collected_links = 0

    metadata = {"scores": []}
    try:
        login(page, email, password)
        while n_collected_links < config.num_scores:
            page.goto(NEXT_URL)

            page.wait_for_selector("div[class='OAIWc']")
            scores = page.locator("div[class='OAIWc']")
            n = min(scores.count(), config.num_scores - n_collected_links)
            for i in range(n):
                div = scores.nth(i)
                a = div.locator("a").first
                link = a.get_attribute("href")
                if not link:
                    raise ValueError("Link not found for score")
                full_link = urljoin(LOGIN_URL, link)
                meta = div.locator("span.AZHap").inner_text().split("•")
                meta = [m.strip() for m in meta]
                score_metadata = parse_metadata(meta, full_link)
                queue.put(score_metadata)
                metadata["scores"].append(score_metadata)
                logger.info(
                    "head queued score %s/%s: %s",
                    n_collected_links + i + 1,
                    config.num_scores,
                    full_link,
                )
            n_collected_links += n
            current_page += 1
            NEXT_URL = SHEET_MUSIC_URL + "?page=" + str(current_page)
        with open(config.output_dir / "metadata.json", "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=4)
    finally:
        for _ in range(config.num_workers):
            queue.put(None)  # Sentinel to signal workers to stop
        browser.close()
        playwright.stop()


def scraper_worker(
    config: ScraperConfig,
    queue: Queue,
    email: str,
    password: str,
    worker_id: int,
):
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=HEADLESS)
    page = browser.new_page()
    try:
        login(page, email, password)
        next_score = queue.get()
        while next_score is not None:
            metadata = next_score
            link = str(metadata["link"])
            score_id = str(metadata["score_id"])

            os.makedirs(config.output_dir / score_id, exist_ok=True)

            page.goto(link, wait_until="domcontentloaded")
            page.wait_for_selector("button[name='download']")

            page.click("button[name='download']")
            with page.expect_download() as download_info:
                page.click("text=MusicXML")
            xml_zip_path = config.output_dir / f"{score_id}.xml.zip"
            xml_path = config.output_dir / f"{score_id}.xml"
            download_info.value.save_as(xml_zip_path)
            with zipfile.ZipFile(xml_zip_path) as archive:
                xml_member = next(
                    name
                    for name in archive.namelist()
                    if name.lower().endswith((".xml", ".musicxml"))
                    and not name.lower().startswith("meta-inf/")
                )
                with archive.open(xml_member) as src, open(xml_path, "wb") as dst:
                    dst.write(src.read())
            os.remove(xml_zip_path)

            convert_xml2abc(
                file_to_convert=str(xml_path),
                output_directory=str(config.output_dir / score_id),
            )

            page.goto(link, wait_until="domcontentloaded")
            page.wait_for_selector("button[name='download']")

            page.click("button[name='download']")
            with page.expect_download() as download_info:
                page.click("text=PDF")
            download_info.value.save_as(config.output_dir / f"{score_id}.pdf")
            images = convert_from_path(config.output_dir / f"{score_id}.pdf")
            for i, img in enumerate(images):
                img.save(config.output_dir / score_id / f"page_{i + 1}.png", "PNG")
            os.remove(config.output_dir / f"{score_id}.pdf")
            os.remove(config.output_dir / f"{score_id}.xml")
            logger.info("worker worker-%s scraped %s", worker_id, score_id)
            next_score = queue.get()
    finally:
        browser.close()
        playwright.stop()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config = parse_args()
    # abc_xml_converter parses global sys.argv internally; clear scraper CLI args first.
    sys.argv = [sys.argv[0]]

    config.output_dir.mkdir(parents=True, exist_ok=True)
    email, password = get_login_credentials()

    with ThreadPoolExecutor(max_workers=config.num_workers + 1) as executor:
        queue: Queue = Queue()
        futures = [executor.submit(head_worker, config, queue, email, password)]
        futures.extend(
            executor.submit(scraper_worker, config, queue, email, password, i + 1)
            for i in range(config.num_workers)
        )
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
