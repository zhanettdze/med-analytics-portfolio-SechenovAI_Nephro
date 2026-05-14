def get_contrast_phases() -> dict:
    return {
        "arterial": {
            "time_seconds": (10, 20),
            "description": "Контраст в почечных артериях",
            "hu_kidney_range": (50, 150),
            "clinical_use": "Оценка артериальной анатомии"
        },
        "parenchymal": {
            "time_seconds": (40, 60),
            "description": "Контраст均匀 в паренхиме",
            "hu_kidney_range": (100, 200),
            "clinical_use": "Выявление опухолей, кист"
        },
        "excretory": {
            "time_seconds": (180, 300),
            "description": "Контраст в чашечно-лоханочной системе",
            "hu_kidney_range": (50, 100),
            "clinical_use": "Оценка оттока мочи, гидронефроз"
        }
    }

def recommend_registration_metric(phase1: str, phase2: str) -> str:
    if phase1 == phase2:
        return "SSD (одинаковые фазы)"
    else:
        return "Mutual Information (разные фазы контраста)"

if __name__ == "__main__":
    phases = get_contrast_phases()
    print("КТ фазы для почек:\n")
    for name, info in phases.items():
        print(f"{name.upper()}:")
        print(f"  Время: {info['time_seconds'][0]}-{info['time_seconds'][1]} сек")
        print(f"  HU: {info['hu_kidney_range']}")
        print(f"  Применение: {info['clinical_use']}\n")
    
    print(f"Рекомендация: {recommend_registration_metric('arterial', 'parenchymal')}")
    print("\nKiTS19 dataset uses contrast-enhanced CT (corticomedullary phase)")
    print("This corresponds to parenchymal phase (HU range 100-200)")