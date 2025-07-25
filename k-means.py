import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def determine_num_clusters(X, max_clusters=8):
    best_k = 2
    best_score = -1
    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
        labels = kmeans.fit_predict(X)
        try:
            score = silhouette_score(X, labels)
        except:
            continue
        if score > best_score:
            best_score = score
            best_k = k
    return best_k

def save_segmentation_as_image(labels_2d, output_path, k_opt):
    segmented_img = (255 * (labels_2d / (k_opt - 1))).astype(np.uint8)
    im = Image.fromarray(segmented_img, mode='L')
    im.save(output_path)

def analyze_kmeans_auto_clusters(image_path, output_folder, max_clusters=8, show=False):
    img = Image.open(image_path).convert('L')
    img.thumbnail((128, 128))

    img_array = np.array(img)
    X = img_array.reshape(-1, 1) / 255.0

    k_opt = determine_num_clusters(X, max_clusters=max_clusters)
    kmeans = KMeans(n_clusters=k_opt, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(X)
    labels_2d = labels.reshape(img_array.shape)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}_segmented.png")

    save_segmentation_as_image(labels_2d, output_path, k_opt)

    if show:
        import matplotlib.pyplot as plt
        plt.imshow(labels_2d, cmap='tab10')
        plt.title(f"{base_name} (k={k_opt})")
        plt.axis('off')
        plt.show()

    return os.path.basename(image_path), k_opt

def process_folder_auto_clusters(folder_path, max_clusters=8, show=False, save_csv=True, csv_name="cluster_results.csv"):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    
    output_folder = os.path.join(folder_path, "segmented")
    os.makedirs(output_folder, exist_ok=True)

    csv_full_path = os.path.join(folder_path, csv_name)

    if save_csv and not os.path.exists(csv_full_path):
        with open(csv_full_path, 'w', encoding='utf-8') as f:
            f.write("file,n_clusters\n")

    for i, file in enumerate(files, 1):
        path = os.path.join(folder_path, file)
        try:
            name, k_opt = analyze_kmeans_auto_clusters(path, output_folder, max_clusters=max_clusters, show=show)
            if save_csv:
                with open(csv_full_path, 'a', encoding='utf-8') as f:
                    f.write(f"{name},{k_opt}\n")
            print(f"[{i}/{len(files)}] ✅ Processed: {name} with k={k_opt}")
        except Exception as e:
            print(f"[{i}/{len(files)}] ⚠️ Error processing {file}: {e}")

    print(f"\n✅ Results saved in: {csv_full_path}")

if __name__ == "__main__":
    folder_path = r"your_path_here"
    process_folder_auto_clusters(
        folder_path,
        max_clusters=8,
        show=False,
        save_csv=True,
        csv_name="cluster_results.csv"
    )
