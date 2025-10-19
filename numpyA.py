import numpy as np

print("\n" + "="*50)
print("          1. ЗАГРУЗКА ДАННЫХ")
print("="*50)

# Загрузка данных
data = np.load('src/data.npz')
nodes = data['nodes']
tris = data['tris']

# Индексы (i,j,k) должны быть целочисленными
tris = tris.astype(int)

print("ПАРАМЕТРЫ ДАННЫХ:")
print("─" * 50)
print(f"Количество узлов: {len(nodes)}")
print(f"Количество треугольников: {len(tris)}")
print(f"Форма массива узлов: {nodes.shape}")
print(f"Форма массива треугольников: {tris.shape}")

print("\n" + "="*50)
print("          2. ВЫЧИСЛЕНИЕ ХАРАКТЕРИСТИК")
print("="*50)

print("ВЫЧИСЛЕНИЕ ЦЕНТРОВ ТРЕУГОЛЬНИКОВ...")
# 1. Координаты геометрических центров треугольников
# Получаем координаты вершин для каждого треугольника
triangle_vertices = nodes[tris]  # Эта конструкция работает в NumPy для целочисленных индексов
centers = np.mean(triangle_vertices, axis=1)

print("ВЫЧИСЛЕНИЕ ПЛОЩАДЕЙ...")
# Площади треугольников через векторное произведение
vec1 = triangle_vertices[:, 1] - triangle_vertices[:, 0]
vec2 = triangle_vertices[:, 2] - triangle_vertices[:, 0]
cross_products = vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0]
areas = 0.5 * np.abs(cross_products)
area_min, area_max = np.min(areas), np.max(areas)

print("ВЫЧИСЛЕНИЕ ДЛИН СТОРОН...")
# Длины сторон треугольников
# Векторизованный расчет длин сторон
v0 = triangle_vertices[:, 0]
v1 = triangle_vertices[:, 1]
v2 = triangle_vertices[:, 2]

side1 = np.linalg.norm(v1 - v0, axis=1)
side2 = np.linalg.norm(v2 - v1, axis=1)
side3 = np.linalg.norm(v0 - v2, axis=1)

side_lengths = np.column_stack((side1, side2, side3))
side_min, side_max = np.min(side_lengths), np.max(side_lengths)

print("ВЫЧИСЛЕНИЕ УГЛОВ...")
# Углы треугольников через теорему косинусов
angles_list = []
for i in range(len(triangle_vertices)):
    a, b, c = side1[i], side2[i], side3[i]

    # Углы по теореме косинусов
    angle_a = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
    angle_b = np.arccos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c))
    angle_c = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))

    angles_list.extend([angle_a, angle_b, angle_c])
angles = np.array(angles_list)
angle_min, angle_max = np.min(angles), np.max(angles)

print("ВЫЧИСЛЕНИЕ МЕТРИКИ L_RMS...")
# Метрика L_rms = sqrt((1/3) * sum(L_i^2))
l_rms = np.sqrt(np.mean(side_lengths ** 2, axis=1))
l_rms_min, l_rms_max = np.min(l_rms), np.max(l_rms)

print("АНАЛИЗ СТАТИСТИКИ ПЛОЩАДЕЙ...")
# Доля треугольников нестандартной площади
# Используем медиану и межквартильный размах для определения нестандартных значений
median_area = np.median(areas)
q1 = np.percentile(areas, 25)
q3 = np.percentile(areas, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

non_standard_mask = (areas < lower_bound) | (areas > upper_bound)
non_standard_ratio = np.sum(non_standard_mask) / len(areas)

print("ПРОВЕРКА ОРИЕНТАЦИИ...")
# 3. Проверка ориентации вершин (по/против часовой стрелки)
# Знак векторного произведения показывает ориентацию
orientations = np.sign(cross_products)
consistent_orientation = np.all(orientations == orientations[0])

print("\n" + "="*50)
print("          3. СТАТИСТИКА ТРЕУГОЛЬНОЙ СЕТКИ")
print("="*50)

print("ОСНОВНЫЕ ХАРАКТЕРИСТИКИ:")
print("─" * 50)

print("ГЕОМЕТРИЧЕСКИЕ ПАРАМЕТРЫ:")
print("─" * 50)
print(f"Площади:           min={area_min:.6f}, max={area_max:.6f}")
print(f"Длины сторон:      min={side_min:.6f}, max={side_max:.6f}")
print(f"Углы (радианы):    min={angle_min:.6f}, max={angle_max:.6f}")
print(f"Метрика L_rms:     min={l_rms_min:.6f}, max={l_rms_max:.6f}")

print("\nАНАЛИЗ КАЧЕСТВА СЕТКИ:")
print("─" * 50)
print(f"Медиана площадей:            {median_area:.6f}")
print(f"Межквартильный размах:       {iqr:.6f}")
print(f"Нижняя граница выбросов:     {lower_bound:.6f}")
print(f"Верхняя граница выбросов:    {upper_bound:.6f}")
print(f"Доля нестандартных площадей:   {non_standard_ratio:.4f}")
print(f"Одинаковая ориентация:       {consistent_orientation}")

print("\n" + "="*50)
print("          4. ДОПОЛНИТЕЛЬНАЯ АНАЛИТИКА")
print("="*50)

print("РАСПРЕДЕЛЕНИЕ ПЛОЩАДЕЙ:")
print("─" * 50)

# Анализ распределения площадей
area_stats = {
    'Минимум': area_min,
    'Максимум': area_max,
    'Среднее': np.mean(areas),
    'Медиана': median_area,
    'Стандартное отклонение': np.std(areas)
}

for stat, value in area_stats.items():
    print(f"  {stat}: {value:.6f}")

print("\nРАСПРЕДЕЛЕНИЕ УГЛОВ:")
print("─" * 50)

# Анализ распределения углов (в градусах для наглядности)
angles_deg = np.degrees(angles)
angle_stats = {
    'Минимум (градусы)': np.min(angles_deg),
    'Максимум (градусы)': np.max(angles_deg),
    'Среднее (градусы)': np.mean(angles_deg)
}

for stat, value in angle_stats.items():
    print(f"  {stat}: {value:.2f}°")

print("\nКЛАССИФИКАЦИЯ ТРЕУГОЛЬНИКОВ:")
print("─" * 50)

# Классифицируем треугольники по качеству на основе отношения сторон
max_sides = np.max(side_lengths, axis=1)
min_sides = np.min(side_lengths, axis=1)
aspect_ratios = max_sides / min_sides

good_triangles = aspect_ratios <= 2.0
fair_triangles = (aspect_ratios > 2.0) & (aspect_ratios <= 5.0)
poor_triangles = aspect_ratios > 5.0

print(f"Хорошие (отношение ≤ 2.0):{np.sum(good_triangles):>4} треугольников ({np.mean(good_triangles)*100:.1f}%)")
print(f"Удовлетворительные (2-5): {np.sum(fair_triangles):>4} треугольников ({np.mean(fair_triangles)*100:.1f}%)")
print(f"Плохие (отношение > 5.0): {np.sum(poor_triangles):>4} треугольников ({np.mean(poor_triangles)*100:.1f}%)")
