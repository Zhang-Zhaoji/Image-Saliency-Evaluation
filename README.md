# Video Saliency Evaluation

```
                                                                                                   
     _______    ___      ___  ________   ___        ___  ___   ________   _________   ___   ________   ________  
    |\  ___ \  |\  \    /  /||\   __  \ |\  \      |\  \|\  \ |\   __  \ |\___   ___\|\  \ |\   __  \ |\   ___  \  
    \ \   __/| \ \  \  /  / /\ \  \|\  \\ \  \     \ \  \\\  \\ \  \|\  \\|___ \  \_|\ \  \\ \  \|\  \\ \  \\ \  \   
     \ \  \_|/__\ \  \/  / /  \ \   __  \\ \  \     \ \  \\\  \\ \   __  \    \ \  \  \ \  \\ \  \\\  \\ \  \\ \  \  
      \ \  \_|\ \\ \    / /    \ \  \ \  \\ \  \____ \ \  \\\  \\ \  \ \  \    \ \  \  \ \  \\ \  \\\  \\ \  \\ \  \   
       \ \_______\\ \__/ /      \ \__\ \__\\ \_______\\ \_______\\ \__\ \__\    \ \__\  \ \__\\ \_______\\ \__\\ \__\  
        \|_______| \|__|/        \|__|\|__| \|_______| \|_______| \|__|\|__|     \|__|   \|__| \|_______| \|__| \|__|  
                                                                                                   
    self-defined useful comparasion functions, designed based on Kummerer, M., Wallis, T. S., & Bethge, M. (2018).   
    Saliency benchmarking made easy: Separating models, maps and metrics. ECCV (pp. 770-787).      
    the full code would consider image1 is the pred image, while the image2 is the ground truth image. Using carefully!   
                                                                                                   
```

It is a **non-official numpy&opencv** implementation of [Kummerer M, Wallis T S A, Bethge M. Saliency benchmarking made easy: Separating models, maps and metrics[C]//Proceedings of the European Conference on Computer Vision (ECCV). 2018: 770-787.](https://arxiv.org/abs/1704.08615 "arxiv article")

**All** code is documented in the evaluation.py file, containing NSS, IG, CC KL-Div & SIM for single image iteration.

**Well annotated!**
