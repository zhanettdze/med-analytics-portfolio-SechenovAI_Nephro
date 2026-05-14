# 04_image_registration

## Содержание

| Файл | Описание |
|:---|:---|
| `01_rigid_registration.py` | Жесткая регистрация (6 параметров: 3 поворота + 3 сдвига) |
| `02_affine_registration.py` | Аффинная регистрация (12 параметров: + масштаб + скос) |
| `03_bspline_registration.py` | Нелинейная регистрация на основе B-spline (сетка контрольных точек) |
| `04_demons_registration.py` | Нелинейная регистрация на основе оптического потока (Thirion's demons) |
| `05_optimization_benchmark.py` | Сравнение оптимизаторов: RegularStepGradientDescent, GradientDescent, ConjugateGradient |
| `06_registration_pipeline.py` | Полный пайплайн, объединяющий все методы регистрации |

## Результаты тестирования (self-test на KiTS19 case_00000)

| Метод | Время | Финальная метрика (MI) | Поле деформации |
|:---|:---|:---|:---|
| Rigid | 3.85 сек | -0.0607 | нет |
| Affine | 0.58 сек | 0.0000 | нет |
| B-spline | 626 сек | -0.0370 | есть (1029 параметров) |
| Demons | 0.21 сек | 0.0000 (RMS) | есть |

## Сравнение оптимизаторов (self-test)

| Оптимизатор | Время | Метрика | Итерации |
|:---|:---|:---|:---|
| RegularStepGradientDescent | 0.68 сек | 0.0000 | 2 |
| GradientDescent | 4.86 сек | -0.0653 | 9 |
| ConjugateGradient | 4.84 сек | -0.0653 | 9 |

**Рекомендация для проекта:** RegularStepGradientDescent (быстрее и стабильнее).

## Запуск

### Отдельные методы

```bash
python3 01_rigid_registration.py
python3 02_affine_registration.py
python3 03_bspline_registration.py
python3 04_demons_registration.py
python3 05_optimization_benchmark.py