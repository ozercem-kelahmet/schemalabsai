# Early stop kısmını değiştir - sadece bu kısım

        # Early stop - %99 hedef
        if best_acc >= 99.0:
            print(f"    🎉 %99+ reached!")
            break
        
        if no_improve >= patience * 2:  # Daha uzun bekle
            print(f"    ⚠️ Early stop at {best_acc:.1f}%")
            break
