import scrapy
import os
from pprint import pprint
base_url = r'http://www.poi86.com/poi/amap/district/'

class POISpider(scrapy.Spider):
    name = 'poi'
    allowed_domains = ['poi86.com', 'www.poi86.com']
    
    region_code_all = [510104, 510105, 510106, 510107, 510108, 510112, 510124]
    region_code = [510104]
    start_urls = [os.path.join(base_url, str(code)+'/1.html') 
                    for code in region_code]


    def parse(self, response):
        head = response.selector.xpath('//h1/text()').extract()
        num_pages = response.xpath('//li[@class="disabled"]/a/text()').extract()
        #Out[21]: ['上一页', '1/2867']
        num_pages = num_pages[-1].split('/')[-1]
        poi_urls = response.xpath('//table[contains(@class, "table-hover")]//a/@href').extract()
        poi_urls = self.url_complete(poi_urls)
        next_page = response.xpath('//a[contains(text(), "下一页")]/@href').extract()
        next_page = self.url_complete(next_page)
        
        pprint(poi_urls[:10])
        for k in range(int(num_pages)):
            yield scrapy.Request(next_page[0], callback=self.parse)

    def url_complete(self, rail):
        main_url = r'http://www.poi86.com'
        return [main_url + x for x in rail]