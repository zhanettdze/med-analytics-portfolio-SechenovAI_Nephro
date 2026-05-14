import time
import numpy as np

np.random.seed(42)

print("=" * 60)
print("Broadcasting в NumPy для обработки КТ почек")
print("=" * 60)

ct_volume = np.random.rand(100, 512, 512)
kidney_mask = np.random.rand(512, 512) > 0.7

print(f"\nФорма КТ: {ct_volume.shape}, форма маски: {kidney_mask.shape}")

print("\n1. Применение маски ко всем срезам")
start = time.perf_counter()
result_loop = np.zeros_like(ct_volume)
for i in range(ct_volume.shape[0]):
    result_loop[i] = ct_volume[i] * kidney_mask
time_loop = time.perf_counter() - start

start = time.perf_counter()
result_bcast = ct_volume * kidney_mask
time_bcast = time.perf_counter() - start
print(f"Цикл: {time_loop:.4f} сек, Broadcasting: {time_bcast:.4f} сек, Ускорение: ~{time_loop / time_bcast:.0f}x")

print("\n2. Нормализация срезов (вычитание среднего)")
start = time.perf_counter()
norm_loop = np.zeros_like(ct_volume)
for i in range(ct_volume.shape[0]):
    norm_loop[i] = ct_volume[i] - ct_volume[i].mean()
time_loop = time.perf_counter() - start

start = time.perf_counter()
mean_vals = ct_volume.mean(axis=(1, 2), keepdims=True)
norm_bcast = ct_volume - mean_vals
time_bcast = time.perf_counter() - start
print(f"Цикл: {time_loop:.4f} сек, Broadcasting: {time_bcast:.4f} сек, Ускорение: ~{time_loop / time_bcast:.0f}x")

real_slices, real_size = 300, 512
real_voxels = real_slices * real_size * real_size
test_voxels = 100 * 512 * 512
time_real = time_bcast * (real_voxels / test_voxels)

print("\n" + "=" * 60)
print("ВЫВОД ДЛЯ ПРОЕКТА")
print("=" * 60)
print(f"""
Для реального КТ почки ({real_slices}×{real_size}×{real_size} = {real_voxels:,} вокселов):
- Нормализация срезов: ~{time_real:.2f} сек (клинически приемлемо)
- Применение маски: без копирования данных
""")
print(f"NumPy {np.__version__}")