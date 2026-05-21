# Kamień Papier Nożyce — detekcja gestów w czasie rzeczywistym

Projekt przygotowany na **Marinaria 2026** przez Koło Naukowe Informatyki Politechniki Morskiej w Szczecinie.

Aplikacja webowa do gry w kamień–papier–nożyce z wykrywaniem gestów dłoni przez kamerę w czasie rzeczywistym. Projekt był uruchamiany i testowany na **MacBooku Pro z układem M5 Pro**.

---

## Jak to działa

- **MediaPipe** wykrywa dłoń i wyznacza 21 punktów kości w czasie rzeczywistym
- Na podstawie pozycji czubków palców względem stawów określany jest gest (kamień / papier / nożyce)
- Obraz z kamery przesyłany jest przez serwer Flask jako strumień MJPEG
- Użytkownik gra przeciwko komputerowi, który losuje swój gest
- Dostępny tryb **autoplay** — gra toczy się automatycznie bez klikania

---

## Wymagania

- Python 3.13+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) — menadżer pakietów i środowisk
- Kamera (wbudowana lub zewnętrzna)
- macOS z układem Apple Silicon (testowane) lub Linux / Windows

---

## Instalacja i uruchomienie

### 1. Sklonuj repozytorium

```bash
git clone https://github.com/KNI-PM-Szczecin/rock-paper-scissors.git
cd rock-paper-scissors
```

### 2. Zainstaluj zależności

```bash
uv sync
```

### 3. Pobierz model detekcji dłoni (MediaPipe)

```bash
curl -L -o hand_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
```

### 4. Pobierz dataset treningowy

```bash
uv run python download_dataset.py
```

Pobiera ~200 MB danych z Google / TensorFlow (zbiór Laurence'a Moroneya — 2520 zdjęć dłoni na białym tle, 3 klasy).

### 5. Wytrenuj model

```bash
uv run python train.py
```

Transfer learning na MobileNetV2 (zamrożony backbone, douczana tylko głowica klasyfikatora). Na Apple Silicon (~M1+) zajmuje ok. 2–5 minut. Model zostaje zapisany do `models/rps_model.pth`.

> Model nie jest dołączony do repozytorium — musisz wygenerować go lokalnie.

### 6. Uruchom aplikację

```bash
uv run python app.py
```

Otwórz przeglądarkę pod adresem: **http://127.0.0.1:5000**

---

## Obsługa

| Akcja | Opis |
|---|---|
| **AUTOPLAY** | Gra toczy się automatycznie — odliczanie 3–2–1, odczyt gestu, wynik, repeat |
| **ZAGRAJ RAZ** | Jedna runda ręcznie |
| **NOWA GRA** | Resetuje wynik |

Gesty rozpoznawane przez kamerę:

| Gest | Symbol |
|---|---|
| Kamień | ✊ |
| Papier | 🖐️ |
| Nożyce | ✌️ |

---

## Struktura projektu

```
.
├── app.py                 # serwer Flask + strumień wideo + logika gry
├── train.py               # trening modelu (transfer learning, MobileNetV2)
├── inference.py           # standalone detekcja przez kamerę (bez przeglądarki)
├── download_dataset.py    # skrypt pobierający dataset
├── templates/
│   └── index.html         # interfejs webowy
├── models/                # tu trafia wytrenowany model (.pth)
├── pyproject.toml
└── uv.lock
```

---

## Technologie

- [PyTorch](https://pytorch.org/) + torchvision — trening i inferencja modelu CNN
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) — detekcja dłoni i landmarki
- [OpenCV](https://opencv.org/) — przechwytywanie obrazu z kamery
- [Flask](https://flask.palletsprojects.com/) — serwer webowy i strumień MJPEG
