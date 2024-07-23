from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# define depth limit
depth_limit = 1


def crawl(url, depth, visited_urls=set()):
    # Check if the depth limit has been reached
    if depth > depth_limit:
        print("Depth limit reached")
        return ""

    # Check if the URL has already been visited
    if url in visited_urls:
        print("URL already visited")
        return ""

    # Add the URL to the set of visited URLs
    visited_urls.add(url)

    # ignore social media links
    if any(domain in url for domain in ["twitter.com", "facebook.com", "linkedin.com"]):
        print("Ignoring social media link")
        return ""

    try:
        # Make a request to the website
        r = requests.get(url)
        r.raise_for_status()
        r_html = r.text

        # Create a BeautifulSoup object and specify the parser
        soup = BeautifulSoup(r_html, "html.parser")

        # Find all the text on the page
        text = soup.find_all(text=True)

        # Remove unnecessary whitespace
        output = ""
        blacklist = [
            "[document]",
            "noscript",
            "header",
            "html",
            "meta",
            "head",
            "input",
            "script",
            "style",
        ]

        for t in text:
            if t.parent.name not in blacklist:
                output += "{} ".format(t)

        # Find all the links on the page
        links = [a["href"] for a in soup.find_all("a", href=True)]

        # Recursively crawl the linked pages
        for link in links:
            absolute_url = urljoin(url, link)
            output += crawl(absolute_url, depth + 1, visited_urls)

        return output
    except Exception as e:
        print(f"ERROR while parsing URL: {url} - {e}")
        return ""


def parse_url_and_get_text(url):
    try:
        # Initialize a set to store the visited URLs
        visited_urls = set()
        raw_text = crawl(url, 0, visited_urls)

        # clean the output
        text = " ".join(raw_text.split())
        text = text.replace("\n", " ")

        return text
    except Exception as e:
        print(f"ERROR while parsing URL: {url} - {e}")
        return ""
