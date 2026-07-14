from train import main

UMBRAL_F1 = 0.80

def test_f1_macro_minimo():
    f1 = main()
    assert f1 > UMBRAL_F1, f"F1 macro demasiado bajo: {f1:.4f} (umbral: {UMBRAL_F1})"
    print(f"✅ Test pasado: F1 macro = {f1:.4f} (> {UMBRAL_F1})")

if __name__ == "__main__":
    test_f1_macro_minimo()