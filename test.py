import os
import cv2

def downsample_images(hr_image_dir, lr_image_dir, keepdims=False, scales=[2, 3, 4]):

    supported_img_formats = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2",
                            ".png", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".tif",
                            ".tiff")

    for scale in scales:
        os.makedirs(os.path.join(lr_image_dir, f"X{scale}"), exist_ok=True)

    for filename in os.listdir(hr_image_dir):
        if not filename.endswith(supported_img_formats):
            continue

        name, ext = os.path.splitext(filename)
        hr_img = cv2.imread(os.path.join(hr_image_dir, filename))
        hr_dims = (hr_img.shape[1], hr_img.shape[0])

        hr_img = cv2.GaussianBlur(hr_img, (0, 0), 1, 1)

        for scale in scales:
            fx = fy = 1.0 / scale
            lr_img = cv2.resize(hr_img, (0, 0), fx=fx, fy=fy, 
                               interpolation=cv2.INTER_CUBIC)
            if keepdims:
                lr_img = cv2.resize(lr_img, hr_dims, 
                                  interpolation=cv2.INTER_CUBIC)
            output_path = os.path.join(
                lr_image_dir, 
                f"X{scale}", 
                f"{name}{ext}"
            )
            cv2.imwrite(output_path, lr_img)

if __name__ == "__main__":
    downsample_images(
        hr_image_dir='/data2/users/jiahaolin/CATANet-main/JF_WHU/HR/night/X4',
        lr_image_dir='/data2/users/jiahaolin/CATANet-main/JF_WHU/LR/night',
        keepdims=False,
        scales=[2, 3, 4]
    )