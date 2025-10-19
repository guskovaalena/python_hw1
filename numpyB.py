import numpy as np
from PIL import Image
import os
from itertools import permutations

# Определяем все возможные преобразования для четвертей изображения
transformations = [
    lambda x: x,  # 0: исходное
    lambda x: np.rot90(x, 1),  # 1: поворот на 90°
    lambda x: np.rot90(x, 2),  # 2: поворот на 180°
    lambda x: np.rot90(x, 3),  # 3: поворот на 270°
    lambda x: np.fliplr(x),  # 4: отражение по горизонтали
    lambda x: np.flipud(x),  # 5: отражение по вертикали
    lambda x: np.transpose(x),  # 6: транспонирование
    lambda x: 255 - x  # 7: инвертирование
]


def normalized_cross_correlation(a, b):
    # Проверяем, что массивы не пустые
    if a.size == 0 or b.size == 0:
        return 0

    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)

    # Проверяем, что массивы не постоянные
    if np.all(a_flat == a_flat[0]) or np.all(b_flat == b_flat[0]):
        return 0

    try:
        corr_matrix = np.corrcoef(a_flat, b_flat)
        if corr_matrix.shape == (2, 2):
            correlation = corr_matrix[0, 1]
        else:
            correlation = 0
    except Exception:
        correlation = 0

    if np.isinf(correlation) or np.isnan(correlation):
        return 0

    return correlation


def find_best_alignment(A, B):
    H, W = A.shape

    # Проверяем, что изображение можно разделить на четверти
    if H % 2 != 0 or W % 2 != 0:
        print("Ошибка: изображение нельзя разделить на равные четверти")
        return B

    h, w = H // 2, W // 2

    # Разбиваем на четверти
    A_quarters = [
        A[:h, :w], A[:h, w:],
        A[h:, :w], A[h:, w:]
    ]

    B_quarters = [
        B[:h, :w], B[:h, w:],
        B[h:, :w], B[h:, w:]
    ]

    # Матрица корреляций и информация о преобразованиях
    correlation_matrix = np.zeros((4, 4))
    transformation_info = {}

    # Для каждой пары четвертей находим наилучшее преобразование
    for i in range(4):
        for j in range(4):
            best_correlation = -np.inf
            best_transform_idx = 0
            best_invert = False

            # Перебираем все преобразования
            for transform_idx, transform in enumerate(transformations):
                try:
                    B_transformed = transform(B_quarters[j])

                    # Пробуем обычную и инвертированную версию
                    for invert in [False, True]:
                        if invert:
                            B_final = 255 - B_transformed
                        else:
                            B_final = B_transformed

                        correlation = normalized_cross_correlation(A_quarters[i], B_final)

                        # Используем абсолютное значение корреляции для сравнения
                        if abs(correlation) > abs(best_correlation):
                            best_correlation = correlation
                            best_transform_idx = transform_idx
                            best_invert = invert
                except Exception as e:
                    # Если преобразование не применимо, пропускаем
                    continue

            # Сохраняем абсолютное значение корреляции для поиска перестановки
            correlation_matrix[i, j] = abs(best_correlation) if not np.isinf(best_correlation) else 0
            transformation_info[(i, j)] = (best_transform_idx, best_invert)

    # Находим наилучшую перестановку четвертей с использованием itertools.permutations
    all_permutations = list(permutations(range(4)))

    best_permutation = None
    best_total_correlation = -np.inf

    for perm in all_permutations:
        total_correlation = 0
        for i in range(4):
            total_correlation += correlation_matrix[i, perm[i]]

        if total_correlation > best_total_correlation and not np.isinf(total_correlation):
            best_total_correlation = total_correlation
            best_permutation = perm

    # Если не найдена подходящая перестановка, используем исходную
    if best_permutation is None:
        print("Предупреждение: не найдена подходящая перестановка, используется исходная")
        best_permutation = (0, 1, 2, 3)
    else:
        print(f"Найдена перестановка с общей корреляцией {best_total_correlation:.4f}")

    # Собираем выровненное изображение B
    B_aligned = np.zeros_like(A)
    for i in range(4):
        j = best_permutation[i]

        transform_idx, invert = transformation_info.get((i, j), (0, False))

        try:
            quarter = transformations[transform_idx](B_quarters[j])
            if invert:
                quarter = 255 - quarter

            # Размещаем четверть в соответствующей позиции
            if i == 0:
                B_aligned[:h, :w] = quarter
            elif i == 1:
                B_aligned[:h, w:] = quarter
            elif i == 2:
                B_aligned[h:, :w] = quarter
            elif i == 3:
                B_aligned[h:, w:] = quarter
        except Exception as e:
            # Если не удалось применить преобразование, используем исходную четверть
            if i == 0:
                B_aligned[:h, :w] = B_quarters[j]
            elif i == 1:
                B_aligned[:h, w:] = B_quarters[j]
            elif i == 2:
                B_aligned[h:, :w] = B_quarters[j]
            elif i == 3:
                B_aligned[h:, w:] = B_quarters[j]

    return B_aligned


def decrypt_message(img_array):
    print("\nЗАПУСК ПРОЦЕССА ДЕШИФРОВКИ")
    print("─" * 50)

    # Проверяем, что массив изображения не пустой
    if img_array is None or img_array.size == 0:
        print("Ошибка: пустой массив изображения")
        return None

    best_message = None
    best_contrast = -np.inf

    # Перебираем все пары каналов
    channel_pairs = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

    print("ПЕРЕБОР КАНАЛЬНЫХ ПАР:")
    print("─" * 50)

    for ch1, ch2 in channel_pairs:
        # Проверяем, что каналы существуют
        if len(img_array.shape) < 3 or ch1 >= img_array.shape[2] or ch2 >= img_array.shape[2]:
            continue

        A = img_array[:, :, ch1].astype(np.float32)
        B = img_array[:, :, ch2].astype(np.float32)

        # Выравниваем канал B относительно A
        try:
            print(f"  Выравнивание каналов {ch1} и {ch2}...")
            B_aligned = find_best_alignment(A, B)
        except Exception as e:
            print(f"  Ошибка при выравнивании каналов {ch1} и {ch2}: {e}")
            continue

        # Вычитаем шум и усиливаем контраст
        message = A - B_aligned

        # Усиливаем контраст перед нормализацией
        message_enhanced = message * 2.0  # Увеличиваем контраст

        # Нормализуем результат
        min_val = np.min(message_enhanced)
        max_val = np.max(message_enhanced)

        # Проверяем, что диапазон не нулевой
        if max_val - min_val == 0:
            continue

        message_normalized = (message_enhanced - min_val) / (max_val - min_val) * 255
        message_normalized = message_normalized.astype(np.uint8)

        # Оцениваем контрастность (дисперсию)
        contrast = np.var(message_normalized)

        print(f"  Контрастность для каналов {ch1} и {ch2}: {contrast:.2f}")

        if contrast > best_contrast:
            best_contrast = contrast
            best_message = message_normalized

    if best_message is not None:
        print(f"Наилучшая контрастность достигнута: {best_contrast:.2f}")
    else:
        print("Не удалось найти подходящее сообщение")

    return best_message


print("\n" + "="*50)
print("          1. НАСТРОЙКА АЛГОРИТМА")
print("="*50)

print("ОПРЕДЕЛЕНЫ ПРЕОБРАЗОВАНИЯ:")
print("─" * 50)
transform_names = [
    "Исходное изображение",
    "Поворот на 90°",
    "Поворот на 180°",
    "Поворот на 270°",
    "Отражение по горизонтали",
    "Отражение по вертикали",
    "Транспонирование",
    "Инвертирование цветов"
]

for i, name in enumerate(transform_names):
    print(f"  {i}: {name}")

print("\n" + "="*50)
print("          2. ОБРАБОТКА ИЗОБРАЖЕНИЙ")
print("="*50)

# Основной процесс дешифровки
for i in range(1, 4):
    try:
        print(f"\nОБРАБОТКА ИЗОБРАЖЕНИЯ {i}")
        print("─" * 50)

        # Формируем правильный путь к файлу
        file_path = f'src/img{i}.png'

        # Проверяем существование файла
        if not os.path.exists(file_path):
            print(f"Файл {file_path} не найден")
            continue

        # Загрузка изображения
        print("ЗАГРУЗКА ИЗОБРАЖЕНИЯ...")
        img = Image.open(file_path)
        img_array = np.array(img)

        # Проверяем, что изображение загружено правильно
        if img_array is None:
            print(f"Не удалось загрузить изображение {file_path}")
            continue

        print(f"Загружено изображение: {file_path}")
        print(f"Форма массива: {img_array.shape}")

        # Если изображение имеет альфа-канал, удаляем его
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]
            print("Удален альфа-канал")

        # Дешифровка сообщения
        decrypted_message = decrypt_message(img_array)

        # Проверяем, что сообщение успешно дешифровано
        if decrypted_message is None:
            print(f"Не удалось дешифровать сообщение из {file_path}")
            continue

        # Сохранение результата
        print("СОХРАНЕНИЕ РЕЗУЛЬТАТА...")
        result_img = Image.fromarray(decrypted_message)
        result_img.save(f'decrypted_message_{i}.png')
        print(f"Сообщение из img{i}.png сохранено в decrypted_message_{i}.png")

        # Анализ результата
        print("\nАНАЛИЗ РЕЗУЛЬТАТА:")
        print("─" * 50)
        print(f"Размер сообщения: {decrypted_message.shape}")
        print(f"Диапазон значений: [{decrypted_message.min()}, {decrypted_message.max()}]")
        print(f"Средняя яркость: {decrypted_message.mean():.2f}")

    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА при обработке img{i}.png: {e}")
        import traceback
        traceback.print_exc()

# Итоговая статистика
print("\n" + "="*50)
print("          3. ИТОГОВАЯ СТАТИСТИКА")
print("="*50)

success_count = 0
for i in range(1, 4):
    result_file = f'decrypted_message_{i}.png'
    if os.path.exists(result_file):
        success_count += 1
        print(f"img{i}.png: УСПЕШНО дешифровано")
    else:
        print(f"img{i}.png: НЕ УДАЛОСЬ дешифровать")

print(f"\nОБЩИЙ РЕЗУЛЬТАТ: {success_count}/3 изображений обработано успешно")

if success_count > 0:
    print("Дешифрованные сообщения сохранены в файлы:")
    for i in range(1, 4):
        result_file = f'decrypted_message_{i}.png'
        if os.path.exists(result_file):
            file_size = os.path.getsize(result_file)
            print(f"  {result_file} ({file_size} байт)")
