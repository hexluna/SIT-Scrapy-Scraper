import scrapy
import os
import json
import io
from PyPDF2 import PdfReader
from urllib.parse import urljoin, urlparse

class FirstPageSpider(scrapy.Spider):
    name = 'everything'
    allowed_domains = ['singaporetech.edu.sg']
    start_urls = ['https://www.singaporetech.edu.sg/']

    # Add list of allowed subpaths (excluding root '/')
    allowed_subpaths = ['/']

    def parse(self, response):
        content_type = response.headers.get('Content-Type', b'').decode('utf-8').lower()
        if 'application/pdf' in content_type or response.url.lower().endswith('.pdf'):
            self.handle_pdf_response(response)
            return

        self.save_page(response)

        base_url = urlparse(response.url)
        links = response.css('a::attr(href)').getall()

        for href in links:
            full_url = urljoin(response.url, href)
            parsed_url = urlparse(full_url)

            if parsed_url.netloc == base_url.netloc:
                path = parsed_url.path

                # Always save root page, but only follow links within allowed subpaths
                if response.url == self.start_urls[0] or any(path == sub or path.startswith(sub) for sub in self.allowed_subpaths):
                    if full_url != response.url:
                        yield scrapy.Request(full_url, callback=self.parse)

    def save_page(self, response):
        parsed = urlparse(response.url)
        path = parsed.path.strip('/')
        query = parsed.query.replace('=', '_').replace('&', '_')
        safe_name = 'root' if not path else path.replace('/', '_')
        if query:
            safe_name = f'{safe_name}_{query}'
        filename = f'output/{safe_name}.json'

        os.makedirs('output', exist_ok=True)

        text_lines = [line.strip() for line in response.css('body *::text').getall() if line.strip()]
        desc_divs = ["\n".join(t.strip() for t in div.css('::text').getall() if t.strip()) for div in response.css('div[class*="desc"]')]
        desc_divs = [text for text in desc_divs if text]
        article_texts = ["\n".join(t.strip() for t in article.css('::text').getall() if t.strip()) for article in response.css('article')]
        article_texts = [text for text in article_texts if text]

        data = {
            'url': response.url,
            'title': response.css('title::text').get(),
            'meta': {
                'description': response.css('meta[name="description"]::attr(content)').get(),
                'keywords': response.css('meta[name="keywords"]::attr(content)').get()
            },
            'text_lines': text_lines,
            'desc_divs': desc_divs,
            'article_texts': article_texts,
            'links': [urljoin(response.url, href) for href in response.css('a::attr(href)').getall()],
            'images': response.css('img::attr(src)').getall()
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def handle_pdf_response(self, response):
        parsed = urlparse(response.url)
        path = parsed.path.strip('/')
        query = parsed.query.replace('=', '_').replace('&', '_')
        safe_name = path.replace('/', '_') or 'pdf_root'
        if query:
            safe_name = f'{safe_name}_{query}'
        filename = f'output/{safe_name}.json'

        os.makedirs('output', exist_ok=True)

        try:
            reader = PdfReader(io.BytesIO(response.body))
            lines = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    lines.extend(line.strip() for line in page_text.splitlines() if line.strip())
            text = "\n".join(lines)
        except Exception as e:
            text = f"Error extracting PDF text: {e}"

        data = {
            'url': response.url,
            'type': 'pdf',
            'content': text
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
