# VRS
## Struktura folderów
* [doc](./doc) – dokumentacja
   * [Prezentacja.pdf](./doc/Prezentacja.pdf)– prezentacja systemu 
   * [Praca_dyplomowa_VRS.pdf](./doc/Praca_dyplomowa_VRS.pdf)– dokumentacja systemu 
 * [src](./src) – kod źródłowy systemu VRN
   * [Module.DIDN](./src/Module.DIDN) – sieć wykrywająca broń na obrazach, YOLOv3
   * [Module.DSDN](./src/Module.DSDN) – sieć klasyfikująca audio
   * [Module.FGN](./src/Module.FGN) – sieć wykrywająca przemoc, Flow Gate Network
   * [Module.Main](./src/Module.Main) – główny moduł koordynujący pracą systemu
   * [Module.VRN](./src/Module.VRN) – sieć wykrywająca przemocom, Violence Recognition Network + sieć VGG16
 * [tools](./tools) – zewnętrzne narzędzi wykorzystane podczas powstawania projektu 
 * [docker-compose.yml](./docker-compose.yml) – plik konfiguracyjny, który pozwala na zbudowanie i uruchomienie kontenerów 
## Konfiguracja systemu
[docker-compose.yml](./docker-compose.yml) - plik rozruchowy z opcjami konfiguracji
- `CAM_IP` - adres streamu wideo
- `FGN_ENABLED` - włącz/wyłącz moduł FGN
- `VRN_ENABLED` - włącz/wyłącz moduł FGN
- `DIDN_ENABLED` - włącz/wyłącz moduł FGN
## Uruchamianie systemu
`docker-compose up --build` - uruchomienie kontenererów razem z ich wcześniejszym zbudowaniem    
`localhost:5341`- adres URL systemu logów Seq