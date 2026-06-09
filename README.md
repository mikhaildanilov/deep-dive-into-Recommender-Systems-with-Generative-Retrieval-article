# Deep Dive into Recommender Systems with Generative Retrieval

Воспроизведение и расширение статьи [TIGER: Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) на датасете Amazon Beauty.

## Результаты

Все полученные метрики сохранены в файле [`metrics.md`](metrics.md).

## Требования

Все эксперименты запускаются на [Kaggle](https://kaggle.com) с бесплатным GPU (T4 x2 или P100). Локальный GPU не требуется.

## Данные

Данные Amazon Beauty скачиваются автоматически внутри ноутбуков через `gdown`. Для экспериментов с TIGER LSH и TIGER RQ-VAE дополнительно нужен файл `data_beauty.pt` с предпосчитанными T5-эмбеддингами (см. ноутбук 0 ниже).

## Ноутбуки

Все ноутбуки находятся в папке `notebooks/`. Запускать в указанном порядке.

---

### 0. Препроцессинг (только для TIGER RQ-VAE и TIGER LSH)

**`tiger-preprocessing.ipynb`**
- Акселератор: **None** (только CPU)
- Генерирует `data_beauty.pt` с T5-эмбеддингами для всех айтемов
- После запуска: скачать `data_beauty.pt` из вкладки Output и загрузить как Kaggle Dataset (например, с именем `data-beauty`)

---

### 1. Бейзлайны - leave-one-out

**`1-data-fast-baselines_loo.ipynb`**
- Акселератор: **None** для EASE и MF-BPR, **GPU T4 x2** для SASRec и BERT4Rec (менять между группами ячеек, комментарии в ноутбуке)
- Запускает: EASE, MF-BPR (3 сида), SASRec (3 сида), BERT4Rec (3 сида)

---

### 2. TIGER (RQ-VAE) - leave-one-out

**`2-tiger-rqvae-loo.ipynb`**
- Акселератор: **GPU T4 x2**
- Требует: `data_beauty.pt`, загружен как Kaggle Dataset (из ноутбука 0)
- Перед запуском: заменить `<YOUR PATH>` в ячейке копирования данных на путь к своему датасету и <YOUR WANDB API KEY> в последней ячейке на свой ключ (или поставить false, если не нужно)

---

### 3. TIGER Random - leave-one-out и temporal

**`3-tiger-random-loo.ipynb`** - leave-one-out сплит  
**`3-tiger-random-temporal.ipynb`** - temporal сплит

- Акселератор: **GPU T4 x2**
- Дополнительных зависимостей нет

---

### 4. TIGER LSH - leave-one-out и temporal

**`4-tiger-lsh-loo.ipynb`** - leave-one-out сплит  
**`4-tiger-lsh-temporal.ipynb`** - temporal сплит

- Акселератор: **GPU T4 x2**
- Требует: `data_beauty.pt`, загружен как Kaggle Dataset (из ноутбука 0)
- Перед запуском: заменить `<YOUR PATH>` на путь к своему датасету

---

### 5. Аблации (RQ3, RQ4) - leave-one-out и temporal

**`5-ablations-loo.ipynb`** - leave-one-out сплит  
**`5-ablations-temporal.ipynb`** - temporal сплит

- Акселератор: **None** или **GPU T4 x2**
- Дополнительных зависимостей нет
- Запускает: TIGER Random с `n_layers` из {2, 3, 4} и `codebook_size` из {64, 256, 1024}, по 5000 шагов каждый

---

## Примечания

- Ноутбуки 3–5 с temporal сплитом клонируют ветку `global_temp_split`, в которой лежит темпорально разбитый `sequential_data.txt` для Amazon Beauty.
- Результаты EASE детерминированы (random seed не используется).
- Все метрики считаются по полному каталогу айтемов без negative sampling.
