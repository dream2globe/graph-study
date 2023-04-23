from pathlib import Path

from src.utils.graph import html2ani

if __name__ == "__main__":
    html_path = Path.cwd() / "data" / "visualization" / "html"
    save_file = Path.cwd() / "data" / "visualization" / "ani" / "ani.gif"
    html2ani(html_path, save_file)
