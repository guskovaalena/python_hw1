import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 1. Определяем функцию F(x, y)
def F(x, y):
    return (np.sin(x ** 2 + y) * np.cos(x - y ** 2) +  # тригонометрические + композиция
            np.exp(-0.1 * (x - np.pi) ** 2) * np.log(1 + abs(y - np.pi)) +  # экспонента + логарифм
            (x ** 2 - y ** 2) / (10 + x * y) +  # полиномиальные + арифметика
            np.tan(x * y) / (1 + x ** 2 + y ** 2))  # тригонометрические + композиция


print("\n" + "="*50)
print("          1. ОПРЕДЕЛЕНИЕ ФУНКЦИИ И ОБЛАСТИ")
print("="*50)

print("ФУНКЦИЯ:")
print("─" * 50)
print("F(x, y) = sin(x² + y) * cos(x - y²) + e^(-0.1(x-π)²) * ln(1 + |y-π|) + (x² - y²)/(10 + xy) + tan(xy)/(1 + x² + "
      "y²)")

# 2. Определяем область [a, b] × [c, d], включающую π
a, b = np.pi / 3, 2 * np.pi  # [π/3, 2π]
c, d = -np.pi / 2, 3 * np.pi / 2  # [-π/2, 3π/2]

print("\nОБЛАСТЬ ОПРЕДЕЛЕНИЯ:")
print("─" * 50)
print(f"x ∈ [π/3, 2π] ≈ [{a:.3f}, {b:.3f}]")
print(f"y ∈ [-π/2, 3π/2] ≈ [{c:.3f}, {d:.3f}]")

# 3. Создаем сетку для вычислений
print("\n" + "="*50)
print("          2. ПОДГОТОВКА ДАННЫХ")
print("="*50)

x = np.linspace(a, b, 500)
y = np.linspace(c, d, 500)
X, Y = np.meshgrid(x, y)

print("ПАРАМЕТРЫ СЕТКИ:")
print("─" * 30)
print(f"Количество точек по x: {len(x)}")
print(f"Количество точек по y: {len(y)}")
print(f"Общее количество точек: {len(x) * len(y):,}")

# 4. Вычисляем значения функции
print("\n" + "="*50)
print("          3. ВЫЧИСЛЕНИЕ ФУНКЦИИ")
print("="*50)

print("ВЫЧИСЛЕНИЕ ЗНАЧЕНИЙ ФУНКЦИИ...")
Z = F(X, Y)

print("\nСТАТИСТИКА ФУНКЦИИ:")
print("─" * 50)
print(f"Минимальное значение: {Z.min():.6f}")
print(f"Максимальное значение: {Z.max():.6f}")
print(f"Среднее значение: {Z.mean():.6f}")
print(f"Стандартное отклонение: {Z.std():.6f}")
print(f"Медиана: {np.median(Z):.6f}")

# 5. Создаем график с настройками
print("\n" + "="*50)
print("          4. ПОСТРОЕНИЕ ГРАФИКА")
print("="*50)

print("СОЗДАНИЕ ГРАФИКА...")
fig, ax = plt.subplots(figsize=(12, 9))

# 6. Отображаем функцию цветом с использованием pcolormesh для лучшего контроля
im = ax.pcolormesh(X, Y, Z,
                   shading='auto',
                   cmap='viridis',
                   vmin=np.percentile(Z, 5),  # Обрезаем выбросы для лучшей визуализации
                   vmax=np.percentile(Z, 95))

# 7. Настраиваем оси и засечки
ax.set_xlabel('$x$', fontsize=14, labelpad=10)
ax.set_ylabel('$y$', fontsize=14, labelpad=10)
ax.set_title(
    '$F(x, y) = \\sin(x^2 + y)\\cos(x - y^2) + e^{-0.1(x-\\pi)^2}\\ln(1 + |y-\\pi|) + \\frac{x^2 - y^2}{10 + xy} + '
    '\\frac{\\tan(xy)}{1 + x^2 + y^2}$',
    fontsize=12, pad=20)

# 8. Устанавливаем засечки с π в tex-нотации
# Для оси X: [π/3, π/2, π, 3π/2, 2π]
x_ticks = [np.pi / 3, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
x_tick_labels = [r'$\frac{\pi}{3}$', r'$\frac{\pi}{2}$', r'$\pi$',
                 r'$\frac{3\pi}{2}$', r'$2\pi$']
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_tick_labels, fontsize=12)

# Для оси Y: [-π/2, 0, π/2, π, 3π/2]
y_ticks = [-np.pi / 2, 0, np.pi / 2, np.pi, 3 * np.pi / 2]
y_tick_labels = [r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$',
                 r'$\pi$', r'$\frac{3\pi}{2}$']
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_tick_labels, fontsize=12)

# 9. Добавляем colorbar с ручными засечками
cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20, pad=0.05)

# Определяем засечки для colorbar на основе распределения данных
z_min, z_max = np.percentile(Z, [5, 95])  # Используем 5-й и 95-й процентили для устойчивости
cbar_ticks = np.linspace(z_min, z_max, 7)  # 7 равномерно распределенных засечек
cbar.set_ticks(cbar_ticks)

# Форматируем подписи colorbar
cbar.ax.tick_params(labelsize=11)
cbar.formatter = ticker.FormatStrFormatter('%.2f')
cbar.update_ticks()

cbar.set_label('$F(x, y)$', fontsize=14, rotation=0, ha='left', va='center', y=1.05)

# 10. Настраиваем сетку для лучшей читаемости
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# 11. Устанавливаем равное масштабирование осей
ax.set_aspect('equal')

# 12. Показываем график
print("ОТОБРАЖЕНИЕ ГРАФИКА...")
plt.show()

# Дополнительная аналитика
print("\n" + "="*50)
print("          5. АНАЛИТИКА ФУНКЦИИ")
print("="*50)

print("РАСПРЕДЕЛЕНИЕ ЗНАЧЕНИЙ ФУНКЦИИ:")
print("─" * 50)

# Вычисляем процентили для анализа распределения
percentiles = [0, 25, 50, 75, 95, 100]
percentile_values = np.percentile(Z, percentiles)

print("Процентили:")
for p, val in zip(percentiles, percentile_values):
    print(f"   {p:2d}%: {val:10.6f}")

print("\nОБЛАСТИ ФУНКЦИИ:")
print("─" * 50)

# Анализируем знаки функции
positive_mask = Z > 0
negative_mask = Z < 0
zero_mask = Z == 0

print(f"Область положительных значений: {positive_mask.sum():,} точек ({positive_mask.sum()/Z.size*100:.1f}%)")
print(f"Область отрицательных значений: {negative_mask.sum():,} точек ({negative_mask.sum()/Z.size*100:.1f}%)")
print(f"Нулевые значения: {zero_mask.sum():,} точек ({zero_mask.sum()/Z.size*100:.1f}%)")

print("\nЭКСТРЕМУМЫ:")
print("─" * 50)

# Находим координаты экстремумов
min_idx = np.unravel_index(np.argmin(Z), Z.shape)
max_idx = np.unravel_index(np.argmax(Z), Z.shape)

print(f"Глобальный минимум: F({X[min_idx]:.3f}, {Y[min_idx]:.3f}) = {Z[min_idx]:.6f}")
print(f"Глобальный максимум: F({X[max_idx]:.3f}, {Y[max_idx]:.3f}) = {Z[max_idx]:.6f}")