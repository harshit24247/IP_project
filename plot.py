import matplotlib.pyplot as plt

def plot_metrics(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'],    label='Train')
    axes[0].plot(history['val_loss'],      label='Val')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(history['train_ang_err'], label='Train')
    axes[1].plot(history['val_ang_err'],   label='Val')
    axes[1].set_title('Angular Error (°)'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(history['acc_5deg'],      label='Acc@5°')
    axes[2].plot(history['acc_10deg'],     label='Acc@10°')
    axes[2].set_title('Accuracy (%)'); axes[2].set_xlabel('Epoch')
    axes[2].legend(); axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png')
    print("📊 Plot saved as training_curves.png")
    plt.show()