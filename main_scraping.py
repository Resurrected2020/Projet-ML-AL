# Import des bibliothèques principales.

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

# Définition du lien commun à toutes les recherches Google Image.

GOOGLE_IMAGE = \
    'https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&'

#_________________________________________________________________

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

#_________________________________________________________________

# Définition du dossier cible.

SAVE_FOLDER = 'images'

#_________________________________________________________________

# Définition de la liste de personnage.

character_list = ['Edward Elric','Alphonse Elric','Gluttony','Roy Mustang','Lust','Winry Rockbell','Scar','Riza Hawkeye','Alex Louis Armstrong','Izumi Curtis','Lelouch Lamperouge','Kururugi Suzaku','C.C.','Kôzuki Karen','Nunnally Lamperouge','Euphemia Li Britannia','Schneizel El Britannia','Milly Ashford','Tôdô Kyoshirô','Jeremiah Gottwald']

#_________________________________________________________________

# main : fonction principale qui permet de scrapper le nombre d'images déterminé par l'utilisateur.

def main():
    number_images = int(input("Combien d'images souhaitez vous scrapper ?"))
    if not os.path.exists(SAVE_FOLDER):
        os.mkdir(SAVE_FOLDER)
    download_images(number_images)

#_________________________________________________________________

# download_images : permet de télécharger le nombre d'images demandé.

def download_images(number_images):
    print('Je cherche...')
    for i, character in enumerate(character_list):
        data = character
        # En fonction de la position dans la character_list, la recherche Google est complétée par le nom de 
        # l'anime
        if i < 10:
            searchurl = GOOGLE_IMAGE + 'q=' + "fullmetal alchemist : brotherhood " + data
        else:
            searchurl = GOOGLE_IMAGE + 'q=' + "code geass " + data


        # Définitions de l'user agent afin de ne pas être rejeté par Google.
        options = Options()
        userAgent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11'
        options.add_argument(f'user-agent={userAgent}')
        # Création de l'objet Selenium pour naviguer sur Chrome.
        browser = webdriver.Chrome(options=options)
        browser.get(searchurl)
        
        # Récupération de tous les éléments qui ont le CSS Selector 'img'
        src_list = []
        results = browser.find_elements_by_css_selector('img')
        # Option qui permet de scroller jusqu'en bas de la page et de cliquer sur le bouton "Page Suivante"
        # si le nombre de résultat obtenu n'est pas suffisant.
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
        # Nous limitons nos résultats finaux avec le nombre d'images demandé.
        results = results[:number_images]
        
        # Option qui permet de scroller sur la page afin de récupérer tous les éléments recherchés.
        for i, result in enumerate(results):
            browser.execute_script("arguments[0].scrollIntoView()", results[i])
            time.sleep(0.1)
            src_link = result.get_attribute('src')
            src_list.append(src_link)
        
        print(f"J'ai trouvé {len(src_list)} images de {data}")
        data = data.replace(" ", "")
        print('Téléchargement en cours...')
        # Les images Google sont sous deux formats de lien : les liens classiques et des liens
        # sous format canva HTML5 en base 64. Nous allons récupérer chaque image et l'enregistrer dans le dossier cible.
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
