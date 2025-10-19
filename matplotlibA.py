import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

# Загрузка данных
print("\n" + "=" * 50)
print("          1. ЗАГРУЗКА ДАННЫХ")
print("=" * 50)

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

# 1. Координаты геометрических центров треугольников
print("\n" + "=" * 50)
print("          2. ВЫЧИСЛЕНИЕ ХАРАКТЕРИСТИК")
print("=" * 50)

print("ВЫЧИСЛЕНИЕ ЦЕНТРОВ ТРЕУГОЛЬНИКОВ...")
triangle_vertices = nodes[tris]  # Получаем координаты вершин для каждого треугольника
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
    # Добавляем проверку для избежания численных ошибок
    cos_a = (b ** 2 + c ** 2 - a ** 2) / (2 * b * c)
    cos_b = (a ** 2 + c ** 2 - b ** 2) / (2 * a * c)
    cos_c = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)

    # Ограничиваем значения косинуса для избежания ошибок округления
    cos_a = np.clip(cos_a, -1.0, 1.0)
    cos_b = np.clip(cos_b, -1.0, 1.0)
    cos_c = np.clip(cos_c, -1.0, 1.0)

    angle_a = np.arccos(cos_a)
    angle_b = np.arccos(cos_b)
    angle_c = np.arccos(cos_c)

    angles_list.extend([angle_a, angle_b, angle_c])
angles = np.array(angles_list)
angle_min, angle_max = np.min(angles), np.max(angles)

print("ВЫЧИСЛЕНИЕ МЕТРИКИ L_RMS...")
# Метрика L_rms = sqrt((1/3) * sum(L_i^2))
l_rms = np.sqrt(np.mean(side_lengths ** 2, axis=1))
l_rms_min, l_rms_max = np.min(l_rms), np.max(l_rms)

print("АНАЛИЗ СТАТИСТИКИ ПЛОЩАДЕЙ...")
# Доля треугольников нестандартной площади
median_area = np.median(areas)
q1 = np.percentile(areas, 25)
q3 = np.percentile(areas, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

non_standard_mask = (areas < lower_bound) | (areas > upper_bound)
non_standard_ratio = np.sum(non_standard_mask) / len(areas)

print("ПРОВЕРКА ОРИЕНТАЦИИ...")
# Проверка ориентации вершин (по/против часовой стрелки)
orientations = np.sign(cross_products)
consistent_orientation = np.all(orientations == orientations[0])

# Вывод результатов
print("\nОСНОВНЫЕ ХАРАКТЕРИСТИКИ:")
print("─" * 50)
print(f"Площади:           min={area_min:.6f}, max={area_max:.6f}")
print(f"Длины сторон:      min={side_min:.6f}, max={side_max:.6f}")
print(f"Углы (радианы):    min={angle_min:.6f}, max={angle_max:.6f}")
print(f"Метрика L_rms:     min={l_rms_min:.6f}, max={l_rms_max:.6f}")

print("\nАНАЛИЗ КАЧЕСТВА СЕТКИ:")
print("─" * 50)
print(f"Медиана площадей:          {median_area:.6f}")
print(f"Межквартильный размах:     {iqr:.6f}")
print(f"Нижняя граница выбросов:   {lower_bound:.6f}")
print(f"Верхняя граница выбросов:  {upper_bound:.6f}")
print(f"Доля нестандартных площадей: {non_standard_ratio:.4f}")
print(f"Одинаковая ориентация:     {consistent_orientation}")


# Функция для поворота точек вокруг заданного центра
def rotate_points(points, angle_degrees, center=(0, 0)):
    angle_rad = np.radians(angle_degrees)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Смещаем точки относительно центра
    points_shifted = points - center

    # Поворачиваем
    x_rotated = points_shifted[:, 0] * cos_angle - points_shifted[:, 1] * sin_angle
    y_rotated = points_shifted[:, 0] * sin_angle + points_shifted[:, 1] * cos_angle

    # Возвращаем к исходной системе координат
    rotated_points = np.column_stack((x_rotated, y_rotated)) + center

    return rotated_points


print("\n" + "=" * 50)
print("          3. ПРЕОБРАЗОВАНИЯ ФИГУР")
print("=" * 50)

# Создаем 4 преобразованные фигуры
figures = []

# Углы поворота вокруг начала координат
global_angles = [90, 180, 270, 360]

print("ВЫПОЛНЕНИЕ ПРЕОБРАЗОВАНИЙ...")
for global_angle in global_angles:
    print(f"  Поворот на {global_angle}°...")

    # 1. Поворачиваем каждый треугольник вокруг своего центра на -90 градусов
    rotated_triangles = []
    for i in range(len(tris)):
        triangle = triangle_vertices[i]
        center = centers[i]
        rotated_triangle = rotate_points(triangle, -90, center)
        rotated_triangles.append(rotated_triangle)

    # 2. Поворачиваем всю фигуру вокруг начала координат
    rotated_figure = []
    for triangle in rotated_triangles:
        rotated_triangle = rotate_points(triangle, global_angle, (0, 0))
        rotated_figure.append(rotated_triangle)

    figures.append(rotated_figure)

print("\n" + "=" * 50)
print("          4. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("=" * 50)

# Нормализуем l_rms для цветовой карты
l_rms_normalized = (l_rms - l_rms_min) / (l_rms_max - l_rms_min)

print("ПОСТРОЕНИЕ ГРАФИКОВ...")

# Визуализация всех четырех фигур
print("СОЗДАНИЕ ОСНОВНОГО ГРАФИКА...")
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for i, (ax, figure, global_angle) in enumerate(zip(axes, figures, global_angles)):
    # Собираем все вершины для отображения точек
    all_vertices = np.vstack(figure)

    # Отображаем точки (вершины треугольников)
    ax.scatter(all_vertices[:, 0], all_vertices[:, 1], color='blue', s=10, alpha=0.6, label='Точки')

    # Отображаем центры треугольников
    # Поворачиваем центры аналогично треугольникам
    rotated_centers = []
    for j in range(len(centers)):
        center_rotated = rotate_points(centers[j].reshape(1, -1), -90, centers[j])
        center_rotated = rotate_points(center_rotated, global_angle, (0, 0))
        rotated_centers.append(center_rotated[0])
    rotated_centers = np.array(rotated_centers)

    ax.scatter(rotated_centers[:, 0], rotated_centers[:, 1], color='red', s=15, alpha=0.8, label='Центры')

    # Отображаем треугольники с цветом, соответствующим L_rms
    triangles_poly = []
    colors = []

    for j, triangle in enumerate(figure):
        triangles_poly.append(triangle)
        colors.append(l_rms_normalized[j])

    collection = PolyCollection(triangles_poly, array=np.array(colors), cmap='viridis',
                                alpha=0.7, edgecolors='black', linewidths=0.5)
    ax.add_collection(collection)

    ax.set_title(f'Фигура {i + 1}: Поворот на {global_angle}°', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    # Используем фиксированное расположение легенды для избежания предупреждения
    ax.legend(loc='upper right', fontsize=9)

# Добавляем colorbar для L_rms с явным указанием осей и увеличенным отступом
cbar = fig.colorbar(collection, ax=axes.tolist(), shrink=0.6, pad=0.08, aspect=30)
cbar.set_label('Нормализованное значение L_rms', fontsize=10)
cbar.ax.tick_params(labelsize=9)

# Используем subplots_adjust с правильными отступами
plt.subplots_adjust(left=0.05, right=0.76, bottom=0.05, top=0.92, wspace=0.25, hspace=0.25)

# Добавляем общий заголовок с увеличенным отступом
fig.suptitle('Четыре преобразованные фигуры с треугольниками', fontsize=16, fontweight='bold', y=0.98)
plt.show()

# Дополнительная визуализация: все фигуры на одном графике
print("СОЗДАНИЕ ОБЪЕДИНЕННОГО ГРАФИКА...")
fig2, ax2 = plt.subplots(figsize=(12, 9))
colors_combined = ['red', 'green', 'blue', 'orange']
markers = ['o', 's', '^', 'D']

# Получаем colormap для использования в цикле
cmap = plt.get_cmap('viridis')

for i, (figure, global_angle, color, marker) in enumerate(zip(figures, global_angles, colors_combined, markers)):
    all_vertices = np.vstack(figure)

    # Отображаем треугольники
    for j, triangle in enumerate(figure):
        # Используем L_rms для цвета через colormap
        triangle_color = cmap(l_rms_normalized[j])

        ax2.fill(triangle[:, 0], triangle[:, 1], color=triangle_color, alpha=0.3)
        ax2.plot(triangle[:, 0], triangle[:, 1], triangle[[0, 2], 0], triangle[[0, 2], 1],
                 color='black', linewidth=0.5, alpha=0.3)

    # Отображаем только несколько точек для легенды
    if i == 0:  # Только для первой фигуры отображаем точки в легенде
        ax2.scatter(all_vertices[0, 0], all_vertices[0, 1], color=color, marker=marker, s=30,
                    label=f'Фигура {i + 1}: Поворот {global_angle}°')

ax2.set_title('Все четыре преобразованные фигуры на одном графике', fontsize=14, fontweight='bold')
ax2.set_xlabel('Координата X', fontsize=12)
ax2.set_ylabel('Координата Y', fontsize=12)
ax2.grid(True, alpha=0.3)

# Используем фиксированное расположение легенды
ax2.legend(loc='upper left', fontsize=10)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=l_rms_min, vmax=l_rms_max))
sm.set_array(l_rms)  # Устанавливаем массив данных
cbar = fig2.colorbar(sm, ax=ax2, shrink=0.8, aspect=20, pad=0.05)
cbar.set_label('L_rms', fontsize=11)
cbar.ax.tick_params(labelsize=9)

ax2.axis('equal')

plt.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()

# Анализ результатов преобразований
print("\n" + "=" * 50)
print("          5. АНАЛИЗ РЕЗУЛЬТАТОВ")
print("=" * 50)

print("СТАТИСТИКА ПРЕОБРАЗОВАННЫХ ФИГУР:")
print("─" * 50)

# Анализ распределения метрик после преобразований
print("РАСПРЕДЕЛЕНИЕ МЕТРИКИ L_RMS:")
print("─" * 50)

l_rms_stats = {
    'Минимум': l_rms_min,
    'Максимум': l_rms_max,
    'Среднее': np.mean(l_rms),
    'Медиана': np.median(l_rms),
    'Стандартное отклонение': np.std(l_rms)
}

for stat, value in l_rms_stats.items():
    print(f"  {stat}: {value:.6f}")

print("\nКЛАССИФИКАЦИЯ ТРЕУГОЛЬНИКОВ ПО L_RMS:")
print("─" * 59)

# Классифицируем треугольники по качеству на основе L_rms
low_quality = l_rms < np.percentile(l_rms, 25)
medium_quality = (l_rms >= np.percentile(l_rms, 25)) & (l_rms <= np.percentile(l_rms, 75))
high_quality = l_rms > np.percentile(l_rms, 75)

print(f"Низкое качество (нижние 25%):    {np.sum(low_quality):>4} треугольников ({np.mean(low_quality) * 100:.1f}%)")
print(
    f"Среднее качество (25%-75%):      {np.sum(medium_quality):>4} треугольников ({np.mean(medium_quality) * 100:.1f}%)")
print(f"Высокое качество (верхние 25%):  {np.sum(high_quality):>4} треугольников ({np.mean(high_quality) * 100:.1f}%)")
