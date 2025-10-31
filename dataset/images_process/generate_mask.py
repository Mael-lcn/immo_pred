import cv2, numpy as np



def mask_CABINET_BLIQUE(img_path):
    out_mask_bin = "mask_auto_pink.png"
    out_mask_feather = "mask_auto_pink_feather.png"

    img = cv2.imread(img_path)[:,:,::-1]  # RGB
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Plage HSV pour rose/magenta — ajuste si nécessaire
    lower1 = np.array([140, 40, 60])   # exemple pour magenta ; éventuellement ajuster
    upper1 = np.array([175, 255, 255])

    # Parfois pink se trouve dans deux plages (split), tu peux ajouter lower2/upper2
    mask = cv2.inRange(hsv, lower1, upper1)

    # Morphologie pour nettoyer
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Save binary mask
    cv2.imwrite(out_mask_bin, mask)

    # Create feathered mask
    mask_f = mask.astype(np.float32)/255.0
    mask_f = cv2.GaussianBlur(mask_f, (51,51), 0)
    cv2.imwrite(out_mask_feather, (np.clip(mask_f,0,1)*255).astype(np.uint8))


def mask_CABINET_BLIQUE():
    print()
