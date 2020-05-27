from icrawler.builtin import BingImageCrawler
import sys
import os
import config

research_word = ['red color', 'blue color', 'yellow color']
max_num = 100
root_dir = config.root_dir + 'tcav/dataset/for_tcav'

for word in research_word:
  path = root_dir + '/' + research_word[0].split(' ')[0]
  crawler = BingImageCrawler(storage = {"root_dir" : path})
  crawler.crawl(keyword = word, max_num = max_num)