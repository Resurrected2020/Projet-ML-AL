import os
import time
import requests # to sent GET requests
import urllib.parse
from base64 import b64decode
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from fake_useragent import UserAgent
import urllib.request

# user can input a topic and a number
# download first n images from google image search

GOOGLE_IMAGE = \
    'https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&'

# The User-Agent request header contains a characteristic string 
# that allows the network protocol peers to identify the application type, 
# operating system, and software version of the requesting software user agent.
# needed for google search
usr_agent = {
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive',
}

SAVE_FOLDER = 'images'

# query = ''
# number_images = ''

character_list = ['Edward Elric','Alphonse Elric','Gluttony','Roy Mustang','Lust','Winry Rockbell','Scar','Riza Hawkeye','Alex Louis Armstrong','Izumi Curtis','Lelouch Lamperouge','Kururugi Suzaku','C.C.','Kôzuki Karen','Nunnally Lamperouge','Euphemia Li Britannia','Schneizel El Britannia','Milly Ashford','Tôdô Kyoshirô','Jeremiah Gottwald']

def main():
    number_images = int(input("Combien d'images souhaitez vous scrapper ?"))
    if not os.path.exists(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)
    download_images(number_images)
    
def download_images(number_images):
    print('Je cherche...')
    for i, character in enumerate(character_list):
        data = character
        # get url query string
        if i < 10:
            searchurl = GOOGLE_IMAGE + 'q=' + "fullmetal alchemist : brotherhood " + data
        else:
            searchurl = GOOGLE_IMAGE + 'q=' + "code geass " + data


        # request url, without usr_agent the permission gets denied
        options = Options()
        # ua = UserAgent()
        userAgent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11'
        # print(userAgent)
        # print(type(userAgent))
        options.add_argument(f'user-agent={userAgent}')
        browser = webdriver.Chrome(options=options)
        browser.get(searchurl)
        
        # find all divs where css_selector='img'
        src_list = []
        results = browser.find_elements_by_css_selector('img')
        while len(results)<number_images:
            results = browser.find_elements_by_css_selector('img')
            browser.execute_script("window.scrollTo(0, document.body.scrollHeight)")
            try:
                button = browser.find_element_by_class_name('frGj1b').click()
                results = browser.find_elements_by_css_selector('img')
                print(len(results))
            except:
                pass
            time.sleep(0.5)
        
        results = results[:number_images]
        # get link from 'src' attribute
        
        # for result in results:
        #     src_link = result.get_attribute('src')
        #     src_list.append(src_link)
        for i, result in enumerate(results):
        #  in range(0,len(results),1):
            browser.execute_script("arguments[0].scrollIntoView()", results[i])
            time.sleep(0.1)
            src_link = result.get_attribute('src')
            src_list.append(src_link)
        
        print(f"J'ai trouvé {len(src_list)} images de {data}")
        data = data.replace(" ", "")
        print('Téléchargement en cours...')
        # get the part of the link encoded in base64 and decode it with base64 module
        for i, src_list_link in enumerate(src_list):
            imagename = SAVE_FOLDER + '/' + data + str(i+1) + '.jpeg'
            if src_list_link.startswith('http'):
                urllib.request.urlretrieve(src_list_link, imagename)
            else:
                link = src_list_link.split(",",1)[1]
                decoded_link = base64.b64decode(link)
                with open(imagename, 'wb') as f:
                    f.write(decoded_link)
                    f.close()
        browser.close()
    print('Et voilà !')

if __name__ == '__main__':
    main()