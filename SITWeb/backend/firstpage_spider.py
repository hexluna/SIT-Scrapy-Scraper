import scrapy

class FirstPageSpider(scrapy.Spider):
    name = 'firstpage'
    start_urls = ['https://www.singaporetech.edu.sg/news/in-the-news?page=1']  # change this

    def parse(self, response):
        text_lines = response.css('body ::text').getall()
        text_clean = '\n'.join(line.strip() for line in text_lines if line.strip())

        yield {
    'url': response.url,
    'title': response.css('title::text').get(),
    'meta': {
        'description': response.css('meta[name="description"]::attr(content)').get(),
        'keywords': response.css('meta[name="keywords"]::attr(content)').get()
    },
    'text_lines': [line.strip() for line in response.css('body *::text').getall() if line.strip()],
    'links': response.css('a::attr(href)').getall(),
    'images': response.css('img::attr(src)').getall()
}