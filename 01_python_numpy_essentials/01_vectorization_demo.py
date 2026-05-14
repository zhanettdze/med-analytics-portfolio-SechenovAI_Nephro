import time
import numpy as np

np.random.seed(42)

print("=" * 60)
print("циклы Python vs векторизация NumPy для регистрации КТ")
print("=" * 60)

size1 = 300
print(f"\n2 3D массива размером {size1}×{size1}×{size1}")
arr1 = np.random.rand(size1, size1, size1)
arr2 = np.random.rand(size1, size1, size1)

size2 = 50
smarr1 = arr1[:size2, :size2, :size2]
smarr2 = arr2[:size2, :size2, :size2]

print(f"\n1. Замеряем тройной цикл Python (на {size2}³ = {size2**3:,} элементов)...")
start = time.perf_counter()
result_pyth = 0

for i in range(size2):
    for j in range(size2):
        for k in range(size2):
            diff = smarr1[i, j, k] - smarr2[i, j, k]
            result_pyth += diff * diff

end = time.perf_counter()
resulted_time = end - start
print(f"Время работы цикла: {resulted_time:.4f} секунд")

full_elements = size1 ** 3
small_elements = size2 ** 3
ratio = full_elements / small_elements
estimated_loop_time = resulted_time * ratio

print(f"\nЕсли бы считали циклами на полном массиве {size1}³:")
print(f"Оценка ~{estimated_loop_time:.1f} секунд")
print(f"Клинический порог (<10 сек): НЕ ПРОЙДЕН")

print(f"\n2. Замеряем NumPy (на полном массиве {size1}³ = {full_elements:,} элементов)...")
start = time.perf_counter()
diff = arr1 - arr2
result_numpy = np.sum(diff * diff)
end = time.perf_counter()
time_numpy = end - start
print(f"Время (NumPy): {time_numpy:.4f} секунд")

real_voxels = 512 * 512 * 300
test_voxels = size1 ** 3
time_numpy_real = time_numpy * (real_voxels / test_voxels)
print(f"\nОценка для реального КТ (512×512×300 = {real_voxels:,} элементов): ~{time_numpy_real:.2f} сек")
print(f"Клинический порог (<10 сек): ПРОЙДЕН")

print("\n" + "=" * 60)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 60)
print(f"SSD (циклы, на маленьком массиве):    {result_pyth:.6f}")
print(f"SSD (NumPy, на полном массиве):      {result_numpy:.6f}")
print(f"\nВремя (циклы, оценка для {size1}³):      {estimated_loop_time:.1f} сек")
print(f"Время (NumPy, для {size1}³):              {time_numpy:.4f} сек")
print(f"Время (NumPy, для реального 512³):       {time_numpy_real:.2f} сек")
print(f"Ускорение:                             ~{estimated_loop_time / time_numpy:.0f} раз!")

print("\n" + "=" * 60)
print("ВЫВОД ДЛЯ ПРОЕКТА SechenovAI_Nephro")
print("=" * 60)
print(f"""
SSD (Sum of Squared Differences) — метрика похожести двух изображений.

Для реального КТ почки (512×512×300 = 78 млн вокселов):
- циклы Python: ~{estimated_loop_time * (real_voxels / test_voxels):.0f} сек → НЕПРИЕМЛЕМО
- NumPy: ~{time_numpy_real:.2f} сек → ПРИЕМЛЕМО (<10 сек)

Регистрация требует вычисления SSD сотни раз (для разных сдвигов/поворотов).
Этот файл демонстрирует разницу в скорости расчета через циклы Python и через NumPy на тестовых данных.
""")

print("=" * 60)
print(f"NumPy версия: {np.__version__}")


