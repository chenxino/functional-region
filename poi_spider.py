# -*- encoding: utf-8 -*-
import requests
from selenium import webdriver
from bs4 import BeautifulSoup
import os
import re
import time
from multiprocessing import Pool, cpu_count, Queue
from lxml import etree, html

def get_page_number(url):
    return url.split('/')[-1].split('.')[0]
    

def url_construct(base, num_pages):
    urls = []
    for k in range(1, int(num_pages) + 1):
        url = os.path.join(os.path.dirname(base), str(k)+'.html')
        urls.append(url)
    return urls


def poi_page_link(source):
    html = BeautifulSoup(source, 'lxml')
    table = html.find_all(class_='table table-bordered table-hover')
    links = table[0].find_all('a')
    links = [x['href'] for x in links]
    return links


def get_poi_urls(k):
    service_args = []
    service_args.append('--load-images=no')
    service_args.append('--disk-cache=yes')
    #options = webdriver.FirefoxOptions()
    #options.add_argument('-headless')
    #options.add_argument('no')
    d = webdriver.PhantomJS(executable_path='./phantomjs', service_args=service_args)
    #d = webdriver.Firefox(executable_path='./geckodriver', firefox_options=options)
    
    session = requests.session()

    STEP = 50
    start = k * STEP + 1
    stop = (k + 1) * STEP + 1

    dest_urls = []
    for region_url in region_urls:
        dest_urls_in_one_region = [region_url + '/' + str(x) + '.html' for x in range(start, stop)]

    dest_urls.extend(dest_urls_in_one_region)

    filename = os.path.join(data_dir, str(k))
    
    if not os.path.exists(filename):
        with open(filename, 'w') as f:
            pass

    for dest_url in dest_urls:
        # dest_url: 'http://www.poi86.com/poi/amap/district/510104/630.html'
        # each url is a page of POI table
        try:
            #print("go to {}".format(dest_url))
            d.get(dest_url) 
            d.implicitly_wait(20)
        except:
            print("fail to connect {} , retrying.".format(dest_url))
            d.get(dest_url)
            d.implicitly_wait(20)
        print("{} responded.".format(dest_url))

        h = etree.fromstring(str(d.page_source))
        #print(h.text)
        print("ElementTree constructed.")
        links = h.xpath('//table[@class="table-hover"]//a/@href')
        print("got {} poi urls".format(len(links)))
        
        for link in links:
            #print(etree.tostring(link, pretty_print=True))
            pass
            #print(link.text)
        #links_ = [x.get_attribute("href") for x in links]   # get 50 POI links in a table"""
        # visite POI detail page and 
        #get_page_source(links_, d)
        

def get_page_source(link_lst, d):
    for url in link_lst:
        try:
            #res = session.get(url)
            d.get(url)
            d.implicitly_wait(20)
            #webdriver.support.wait
        except:
            print("failed to connect {}, retrying.".format(url))
            #res = session.get(url)
            d.get(url)
            d.implicitly_wait(20)

        # h = HTML(res.text)    
        h = etree.HTML(d.page_source)
        print("{} responded(Detail page)".format(url))
        name = h.xpath('//h1/text()')[0]
        category = h.xpath('//li[@class="list-group-item"][6]/text()')[0]
        location = h.xpath('//li[@class="list-group-item"][7]/text()')[0]
        print([name, location, category])
        return [name, location, category]


main_url = r'http://www.poi86.com' 
base_url = r'http://www.poi86.com/poi/amap/district/'
region_code_all = [510104, 510105, 510106, 510107, 510108]
region_urls = [os.path.join(base_url, str(x)) for x in region_code_all]
data_dir = '/home/dlbox/Documents/func_region/Data/POI'

def main():

    ITERATION_COUNT = cpu_count() - 2
    print("total {} CPU cores".format(ITERATION_COUNT))

    useManyCore = True
    ST = time.time()

    if useManyCore:
        q = Queue()
        pool = Pool(processes=ITERATION_COUNT)
        pool.map(get_poi_urls, range(ITERATION_COUNT))
        pool.close()
        pool.join()

    ET = time.time()
    print("used {}s.".format(ET - ST))


if __name__ == '__main__':
    main()