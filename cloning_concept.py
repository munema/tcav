from icrawler.builtin import BingImageCrawler
import sys
import os
import config

research_word = ['violet color', 'black color', 'white color']
max_num = 150
root_dir = config.root_dir + 'tcav/dataset/for_tcav'

for i,word in enumerate(research_word):
  path = root_dir + '/' + research_word[i].split(' ')[0]
  crawler = BingImageCrawler(storage = {"root_dir" : path})
  crawler.crawl(keyword = word, max_num = max_num)