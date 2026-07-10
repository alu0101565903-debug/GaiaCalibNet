import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib

def plot_sample_stars():
    x_path = "data/processed/X.npy"
    y_path = "data/processed/Y.npy"

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print("[!] ERROR: No se encuentran las matrices .npy en data/processed/")
        return

    # Cargar matrices
    X = np.load(x_path)
    Y = np.load(y_path)

    # Configurar el gráfico con dos subplots paralelos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Determine indices to plot (handle small datasets)
    n_samples = X.shape[0]
    if n_samples == 0:
        print("[!] ERROR: Loaded X.npy is empty.")
        return

    idx1 = 0
    idx2 = 1 if n_samples > 1 else 0

    # Graficar la primera estrella (índice idx1)
    ax1.plot(X[idx1], label="Input (X)", color="blue", alpha=0.7, linewidth=1.5)
    ax1.plot(Y[idx1], label="Target (Y)", color="red", alpha=0.7, linewidth=1.5)
    ax1.set_title(f"Estrella de Prueba 1 (Índice {idx1})")
    ax1.set_xlabel("Canal / Píxel (0-300)")
    ax1.set_ylabel("Flujo")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Graficar la segunda estrella (índice idx2)
    ax2.plot(X[idx2], label="Input (X)", color="blue", alpha=0.7, linewidth=1.5)
    ax2.plot(Y[idx2], label="Target (Y)", color="red", alpha=0.7, linewidth=1.5)
    ax2.set_title(f"Estrella de Prueba 2 (Índice {idx2})")
    ax2.set_xlabel("Canal / Píxel (0-300)")
    ax2.set_ylabel("Flujo")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    # Decide whether to show interactively or save based on Matplotlib backend
    backend = matplotlib.get_backend().lower()
    non_interactive_backends = ('agg', 'pdf', 'ps', 'svg', 'cairo')
    if backend not in non_interactive_backends:
        try:
            print("Mostrando gráfico... (Cierra la ventana para terminar el script)")
            plt.show()
            return
        except Exception:
            # Fall through to save if show fails
            pass

    # Save to results/plots or data/processed/plots
    out_dir_candidates = ["results/plots", "data/processed/plots"]
    out_dir = None
    for c in out_dir_candidates:
        try:
            os.makedirs(c, exist_ok=True)
            out_dir = c
            break
        except Exception:
            continue

    if out_dir is None:
        out_dir = "."

    out_path = os.path.join(out_dir, "sample_stars.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Gráfico guardado en: {out_path}")

if __name__ == "__main__":
    plot_sample_stars()