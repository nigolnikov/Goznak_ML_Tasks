# Goznak_ML_Tasks
## Задание 1 Структуры и алгоритмы
Файл `multiplicate.py` содержит реализацию функции `multiplicate(A)`.  
Команда для запуска скрипта:  
`python multiplicate.py [1,2,3,4]`  
Числа в массиве должны быть разделены запятой без пробела. Скрипт выведет результат в командной строке.

## Задание 2 ML\DL
Реализованы 2 модели для решения задач классификации и denoising. Используемы фреймворк - Pytorch.  
Файл `models.py` содержит классы моделей.  
Файл `utils.py` содержит классы для создания датасетов типа `torch.utils.data.Dataset` и вспомогательные функции.

### Классификация
Для решения задачи классификации реализована СНС, архитектуру которой можно изучить в файле `models.py`  
Скрипт для обучения модели - `classification_train.py`  
Скрипт для запуска модели - `classification.py`  
Команда для запуска:  
`python classification.py mel.npy`  
Где `mel.npy` - файл mel-спектрограммы. Скрипт выведет результат классификации.

### Denoising
Для решения задачи denoising реализована U-Net-подобная СНС, архитектуру которой можно изучить в файле `models.py`  
Скрипт для обучения модели - `denoising_train.py`  
Скрипт для запуска модели - `denoising.py`  
Команда для запуска:  
`python denoising.py noisy.npy clean.npy`  
Где `noisy.npy` - файл зашумленной mel-спектрограммы, `clean.npy` - название файла для сохранения обработанной mel-спектрограммы.
