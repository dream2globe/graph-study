from pathlib import Path

import imageio
from PIL import Image
from playwright.sync_api import sync_playwright


def html2ani(source, target, temp: None | str = None):
    if temp is None:
        temp = Path().cwd() / "temp"
    html_files = source.glob("*.html")
    for html in html_files:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False, args=["--start-maximized"])
            page = browser.new_page()
            page.goto(
                f"file://{html}",
                timeout=60000,
                wait_until="networkidle",
            )
            temp_file = temp / f"{html.stem}.png"
            page.screenshot(path=temp_file, full_page=True)
            browser.close()
    # target
    png_files = [str(path) for path in target.glob("*.png")]
    pngs = [Image.open(png) for png in sorted(png_files)]
    imageio.mimsave("test.gif", pngs, fps=2)


if __name__ == "__main__":
    html_path = Path.cwd() / "data" / "visual" / "html"
    ani_path = Path.cwd() / "data" / "visual" / "ani"
    html2ani(html_path, ani_path)
