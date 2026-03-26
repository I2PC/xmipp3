#!/usr/bin/env python3

"""/***************************************************************************
 *
 * Authors:    Daniel Marchan Torres da.marchan@cnb.csic.es
 *
* CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
  ***************************************************************************/
"""

import os
import sys
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import mrcfile
from multiprocessing import Pool, cpu_count
import numpy as np
from scipy.ndimage import gaussian_filter, zoom, rotate, shift, affine_transform
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
from sklearn.manifold import TSNE
from skimage.metrics import structural_similarity as ssim

from xmipp_base import XmippScript


# Constants
LABELS = 'labels'
SCORE = 'score'
FN = "class_representatives"


def create_directory(directory_path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def read_list_from_txt(filename):
    """
    Reads the contents of a .txt file and returns them as a list of strings.

    Parameters:
    filename (str): The name of the file to read.

    Returns:
    list: A list containing each line from the file as a separate item.
    """
    with open(filename, 'r') as file:
        data_list = file.read().splitlines()

    return data_list


def gaussian_blur(image, sigma=1):
    """
    Apply Gaussian blur to the image using SciPy's gaussian_filter.

    :param image: Input image (2D or 3D array).
    :param sigma: Standard deviation for Gaussian kernel. Can be a single value or a sequence for each axis.
    :return: Blurred image.
    """
    return gaussian_filter(image, sigma=sigma)


def z_normalize(image):
    return (image - np.mean(image)) / np.std(image)


def downsample(image, target_size=(128, 128)):
    """
    Downsample the grayscale image using SciPy's zoom to resize the image to the target size.

    :param image: Input grayscale image (2D array).
    :param target_size: Target size as a tuple (height, width).
    :return: Resized grayscale image.
    """
    # Calculate zoom factors for each dimension
    zoom_factors = [target_size[0] / image.shape[0], target_size[1] / image.shape[1]]

    # Apply zoom (resizing)
    return zoom(image, zoom_factors, order=3)  # Cubic interpolation (order=3)


def preprocess_image(image, target_size=(128, 128), apply_gaussian=True):
    """
    Preprocess the input image by normalizing and downsampling.
    """

    if apply_gaussian:
        blurred_image = gaussian_blur(image)
    else:
        blurred_image = image

    downsampled_image = downsample(blurred_image, target_size)
    normalized_image = z_normalize(downsampled_image)

    return normalized_image


def load_and_preprocess_images_from_mrcs(mrcs_file_path, txt_ids_path=None, target_size=(128, 128)):
    """Load images from the specified .mrcs file and preprocess them, returning image names and preprocessed images."""
    images = []
    image_names = []

    if os.path.exists(txt_ids_path):
        image_names.extend(read_list_from_txt(txt_ids_path))

    with mrcfile.open(mrcs_file_path, permissive=True) as mrc:
        # Read the data (it's a stack of 2D images)
        img_stack = mrc.data

        # Iterate over each 2D image in the stack
        for idx, img in enumerate(img_stack):
            # Preprocess the image (resize, normalize, etc.)
            preprocessed_image = preprocess_image(img, target_size=target_size)
            # Append the preprocessed image to the list
            images.append(preprocessed_image)

            if not os.path.exists(txt_ids_path):
                # Create a name for each image based on index or a naming convention
                filename = f"image_{idx + 1}"  # You can modify this as needed
                image_names.append(filename)

    return images, image_names


def compute_ssim(img1, img2):
    return ssim(img1, img2, data_range=img1.max() - img1.min())


def align_images(img1, img2, angle_range=180, coarse_angle_step=20, fine_angle_step=4, shift_range=12, coarse_shift_step=3, fine_shift_step=1):
    best_ssim = -1
    best_angle = 0
    best_shift = (0, 0)

    # Coarse search
    for angle in range(-angle_range, angle_range, coarse_angle_step):
        rotated_img2 = rotate(img2, angle, reshape=False)
        for x_shift in range(-shift_range, shift_range, coarse_shift_step):
            for y_shift in range(-shift_range, shift_range, coarse_shift_step):
                shifted_img2 = shift(rotated_img2, (x_shift, y_shift))
                current_ssim = compute_ssim(img1, shifted_img2)
                if current_ssim > best_ssim:
                    best_ssim = current_ssim
                    best_angle = angle
                    best_shift = (x_shift, y_shift)

    # Fine search around the best coarse result
    for angle in range(best_angle - coarse_angle_step, best_angle + coarse_angle_step + 1, fine_angle_step):
        rotated_img2 = rotate(img2, angle, reshape=False)
        for x_shift in range(best_shift[0] - coarse_shift_step, best_shift[0] + coarse_shift_step + 1, fine_shift_step):
            for y_shift in range(best_shift[1] - coarse_shift_step, best_shift[1] + coarse_shift_step + 1, fine_shift_step):
                shifted_img2 = shift(rotated_img2, (x_shift, y_shift))
                current_ssim = compute_ssim(img1, shifted_img2)
                if current_ssim > best_ssim:
                    best_ssim = current_ssim
                    best_angle = angle
                    best_shift = (x_shift, y_shift)

    aligned_img2 = shift(rotate(img2, best_angle, reshape=False), best_shift)
    transform_params = [best_angle, best_shift]

    # Create transformation matrices (Convention is different that is why the -)
    rotation_matrix = np.array([[np.cos(np.radians(-best_angle)), -np.sin(np.radians(-best_angle)), 0],
                                [np.sin(np.radians(-best_angle)), np.cos(np.radians(-best_angle)), 0],
                                [0, 0, 1]])
    translation_matrix = np.array([[1, 0, best_shift[0]],
                                   [0, 1, -best_shift[1]],
                                   [0, 0, 1]])

    # Combined transformation matrix
    combined_matrix = np.dot(translation_matrix, rotation_matrix)

    return aligned_img2, best_ssim, combined_matrix, transform_params


def transform_image(img, combined_matrix):
    '''
    This function applies a transformation matrix to an image. The transformation matrix is generated during
    alignment and can be used to align particles of different classes in a cluster to compute a new 2D average.
    This function adds a small offset.
    :param img: The image to be transformed.
    :param combined_matrix: The combined rotation and translation matrix.
    :return: The transformed image.
    '''
    # Extract rotation part and translation part from the combined matrix
    rotation_matrix = combined_matrix[:2, :2]
    translation_vector = combined_matrix[:2, 2]

    # Calculate center of the image
    center = np.array(img.shape) / 2
    offset = center - np.dot(rotation_matrix, center) + translation_vector

    # Apply affine transformation with the combined matrix
    transformed_img = affine_transform(img, rotation_matrix, offset=offset, order=1, mode='constant', cval=0.0,
                                       output_shape=img.shape)

    return transformed_img


def transform_image_shift_rotate(img, transform_params):
    '''
    This function applies a transformation functions to an image. The transformation is generated during
    alignment and can be used to align particles of different classes in a cluster to compute a new 2D average (Use this one).
    :param img:
    :param transform_params:
    :return:
    '''
    # Extract rotation part and translation part from the combined matrix
    angle = transform_params[0]
    shifts = transform_params[1]
    transformed_img = shift(rotate(img, angle, reshape=False), shifts)

    return transformed_img


def calculate_ssim_for_pair(args):
    i, j, images, shift_range, coarse_shift_step, fine_shift_step = args
    aligned_img2, best_ssim, _, _ = align_images(images[i], images[j], shift_range=shift_range, coarse_shift_step=coarse_shift_step, fine_shift_step=fine_shift_step)
    return i, j, best_ssim


def build_similarity_matrix(images, cpu_numbers=8):
    n = len(images)
    similarity_matrix = np.zeros((n, n))

    # Compute shift parameters once
    dim = np.shape(images[0])[0]
    shift_range = int(dim * 0.20)
    coarse_shift_step = int(shift_range * 0.25)
    fine_shift_step = int(coarse_shift_step * 0.35)

    indices = [(i, j, images, shift_range, coarse_shift_step, fine_shift_step) for i in range(n) for j in range(i, n)]

    with Pool(cpu_numbers) as pool:
        results = pool.map(calculate_ssim_for_pair, indices)

    for i, j, score in results:
        similarity_matrix[i][j] = score
        similarity_matrix[j][i] = score

    return similarity_matrix


def generate_distance_matrix(similarity_matrix):
    # Convert similarity matrix to distance matrix
    return 1 - similarity_matrix


def apply_pca(distance_matrix, variance=0.95):
    """This function applies PCA with % Variance retention"""
    pca = PCA(n_components=variance)
    pca_transformed = pca.fit_transform(distance_matrix)
    # Get the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    return pca_transformed, explained_variance_ratio


def apply_tsne_2d(data):
    data_size = len(data)
    if data_size <= 10:
        perplexity = 5
    elif data_size <= 20:
        perplexity = 10
    else:
        perplexity = data_size - 10

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(data)
    return tsne_result


def determine_optimal_clusters_kmeans(data, min_clusters=3, max_clusters=10):
    """Determine the optimal number of clusters using K-means clustering."""
    wcss = []
    silhouette_scores = []
    for n in range(min_clusters, max_clusters+1):
        kmeans = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    optimal_clusters = np.argmax(silhouette_scores) + min_clusters

    return optimal_clusters, wcss, silhouette_scores


def determine_optimal_clusters_hierarchical(data, min_clusters=3, max_clusters=10, linkage='ward', metric='euclidean'):
    """Determine the optimal number of clusters using hierarchical clustering."""
    silhouette_scores = []
    for n in range(min_clusters, max_clusters + 1):
        hierarchical = AgglomerativeClustering(n_clusters=n, linkage=linkage, metric=metric)
        labels = hierarchical.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, labels))

    optimal_clusters = np.argmax(silhouette_scores) + min_clusters

    return optimal_clusters, silhouette_scores


def determine_optimal_clusters(data, output_directory, min_clusters=3, max_clusters=10, debug_plots=0):
    """Determine the optimal clustering parameters for K-means and hierarchical."""
    optimal_kmeans, wcss, silhouette_scores_kmeans = determine_optimal_clusters_kmeans(data, min_clusters, max_clusters)
    optimal_hierarchical, silhouette_scores_hierarchical = determine_optimal_clusters_hierarchical(data, min_clusters, max_clusters)

    print(f"Optimal number of clusters for K-means: {optimal_kmeans}")
    print(f"Optimal number of clusters for Hierarchical Clustering: {optimal_hierarchical}")

    if debug_plots:
        plot_optimal_clusters_kmeans(min_clusters, max_clusters, wcss, silhouette_scores_kmeans, output_directory)
        plot_optimal_clusters_hierarchical(min_clusters, max_clusters, silhouette_scores_hierarchical, output_directory)

    return {'kmeans': optimal_kmeans,
        'hierarchical': optimal_hierarchical}


def perform_kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels


def perform_hierarchical_clustering(data, n_clusters):
    agg_clustering =  AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', metric='euclidean')
    labels = agg_clustering.fit_predict(data)
    return labels


def align_single_image(args):
    """Align a single image to the representative image."""
    representative_image, image = args
    aligned_image, score, _, _ = align_images(representative_image, image)
    return aligned_image, score


def align_images_to_representative(representative_image, cluster_images, cores=8):
    """
    Align cluster images to the representative image using multiprocessing.

    Args:
        representative_image: The representative image to align others to.
        cluster_images: List of images in the cluster to align.
        cores: Number of threads to use for parallel processing.

    Returns:
        tuple: (aligned_images, ssim_scores)
    """
    with Pool(cores) as pool:
        # Prepare arguments as tuples of (representative_image, image)
        args = [(representative_image, image) for image in cluster_images]

        # Parallelize the alignment process
        results = pool.map(align_single_image, args)

    # Separate aligned images and SSIM scores from results
    aligned_images, ssim_scores = zip(*results)

    return list(aligned_images), np.array(ssim_scores)


def get_images_to_representative_alignment(labels, images, image_names, vectors, cores):
    results_cluster = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        cluster_indices, = np.nonzero(labels == label)
        cluster_vectors = vectors[cluster_indices]
        cluster_images = images[cluster_indices]
        cluster_image_names = image_names[cluster_indices]
        # Compute centroid of the cluster
        centroid = np.mean(cluster_vectors, axis=0)
        # Find the image closest to the centroid
        closest, _ = pairwise_distances_argmin_min([centroid], cluster_vectors)
        representative_image = cluster_images[closest[0]]

        # Align images to the representative image
        aligned_cluster_images, ssim_scores = align_images_to_representative(representative_image, cluster_images, cores)
        results_cluster[label] = [aligned_cluster_images, cluster_image_names, ssim_scores, cluster_vectors]

    return results_cluster


def sort_images_in_cluster(labels, result_dict, output_dir):
    """
    Sorts images within each cluster based on SSIM scores and saves cluster assignments.

    Args:
        labels: List or array of cluster labels.
        result_dict: Dictionary containing aligned images, image names, SSIM scores, etc. for each cluster.
        output_dir: Directory to save the cluster assignments.

    Returns:
        best_clusters_with_names: Dictionary mapping cluster labels to sorted image names.
        sorted_results: Dictionary mapping cluster labels to sorted image data (images, names, and scores).
    """
    best_clusters_with_names = {}
    sorted_results = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        aligned_images, image_names, ssim_scores, _ = result_dict[label]
        aligned_images = np.array(aligned_images)

        # Sort images by SSIM scores in descending order
        sorted_indices = np.argsort(ssim_scores)[::-1]
        sorted_images = aligned_images[sorted_indices]
        sorted_image_names = image_names[sorted_indices].tolist()
        sorted_ssim_values = np.array(ssim_scores)[sorted_indices]

        # Store sorted names for this cluster
        if label not in best_clusters_with_names:
            best_clusters_with_names[label] = []
        best_clusters_with_names[label].extend(sorted_image_names)

        # Store sorted data for plotting
        sorted_results[label] = (sorted_images, sorted_image_names, sorted_ssim_values)

    # Save cluster assignments to a file
    with open(os.path.join(output_dir, "best_clusters_with_names.txt"), "w") as f:
        for label, names in best_clusters_with_names.items():
            f.write(f"Cluster {label}:\n")
            for name in names:
                f.write(f"\t{name}\n")

    print("Cluster assignments with image names:", best_clusters_with_names)
    return best_clusters_with_names, sorted_results


def extract_array_like_results(result_dict):
    labels = []
    images = []
    vectors = []

    for label, results in result_dict.items():   # results_cluster[label] = [aligned_cluster_images, cluster_image_names, ssim_scores, cluster_vectors]
        aligned_cluster_images, _, _, cluster_vectors = results
        labels.extend([label] * len(cluster_vectors))
        images.extend(aligned_cluster_images)
        vectors.extend(cluster_vectors)

    labels_array = np.array(labels)
    images_array = np.array(images)
    vectors_array = np.array(vectors)

    return labels_array, images_array, vectors_array

# ------------------------------------- PLOTS FUNCTIONS ---------------------------------------------
def plot_similarity_matrix(similarity_matrix, labels=None, output_directory=''):
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap='viridis', xticklabels=labels, yticklabels=labels)
    plt.title('Image Similarity Matrix (SSIM)')
    plt.xlabel('Image Index')
    plt.ylabel('Image Index')
    plot_path = os.path.join(output_directory, 'similarity_matrix.png')
    plt.savefig(plot_path)
    plt.close()


def plot_PCA(cumulative_variance, output_directory):
    import matplotlib.pyplot as plt
    # Optionally, plot the cumulative explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_variance, marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance explained')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_directory, 'pca_cumulative_components.png'))
    plt.close()


def plot_optimal_clusters_kmeans(min_clusters, max_clusters, wcss, silhouette_scores, output_directory):
    import matplotlib.pyplot as plt
    # Elbow Method
    plt.figure(figsize=(10, 6))
    plt.plot(range(min_clusters, max_clusters+1), wcss, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal Clusters (K-means)')
    plt.savefig(os.path.join(output_directory, 'elbow_method_kmeans.png'))
    plt.close()

    # Silhouette Scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(min_clusters, max_clusters+1), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Optimal Clusters (K-means)')
    plt.savefig(os.path.join(output_directory, 'silhouette_scores_kmeans.png'))
    plt.close()


def plot_optimal_clusters_hierarchical(min_clusters, max_clusters, silhouette_scores, output_directory):
    import matplotlib.pyplot as plt
    # Silhouette Scores
    plt.figure(figsize=(10, 6))
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Optimal Clusters (Hierarchical)')
    plt.savefig(os.path.join(output_directory, 'silhouette_scores_hierarchical.png'))
    plt.close()


def image_scatter_plot(vectors, images, labels, output_directory, zoom=0.35,
                       title='t-SNE visualization of clustered images', cluster_type=''):
    """
    Create a scatter plot of images using t-SNE results.
    Displays the images in grayscale with distinct border colors representing their cluster labels.
    Includes a legend for cluster colors.

    Args:
        vectors: 2D array-like (t-SNE or PCA coordinates for images).
        images: List of image arrays (grayscale images expected).
        labels: Array of cluster labels for coloring borders.
        output_directory: Directory to save the resulting plot.
        zoom: Float value to control the size of images in the scatter plot.
        title: Title of the plot.
        cluster_type: Name of the cluster type (used in the output filename).
    """
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(16, 10))

    # Get unique cluster labels and their count
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    # Logic for selecting the colormap
    if num_clusters <= 10:
        cmap = plt.get_cmap('tab10')  # Up to 10 clusters
    elif num_clusters <= 20:
        cmap = plt.get_cmap('tab20')  # Between 11 and 20 clusters
    else:
        # For more than 20 clusters, generate a custom colormap using HSV
        colors = plt.cm.hsv(np.linspace(0, 1, num_clusters))  # Generate distinct colors
        cmap = ListedColormap(colors)

    # Assign distinct colors to each cluster
    cluster_colors = {label: cmap(i / num_clusters) for i, label in enumerate(unique_labels)}

    # Scatter plot (no colormap normalization, uses cluster_colors)
    scatter = ax.scatter(vectors[:, 0], vectors[:, 1], c=[cluster_colors[label] for label in labels], alpha=0.6)

    # Annotate the plot with images
    for xy, img, label in zip(vectors, images, labels):
        # Create the OffsetImage with the 'gray' colormap
        imagebox = OffsetImage(img, cmap='gray', zoom=zoom)
        imagebox.image.axes = ax
        # Set the border color according to the assigned cluster color
        ab = AnnotationBbox(imagebox, xy, frameon=True,
                            bboxprops=dict(edgecolor=cluster_colors[label], lw=2))
        ax.add_artist(ab)

    # Add a legend for cluster labels and their colors
    legend_elements = [
        Patch(facecolor=color, edgecolor=color, label=f'Cluster {label}')
        for label, color in cluster_colors.items()
    ]
    ax.legend(handles=legend_elements, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    # Finalize the plot
    plt.title(title)
    tsne_plot_path = os.path.join(output_directory, cluster_type + '_cluster_visualization_with_images.png')
    plt.savefig(tsne_plot_path, bbox_inches='tight')
    plt.close()
    print(f"t-SNE plot saved to {tsne_plot_path}")


def plot_individual_clusters(sorted_results, output_dir, max_images_per_cluster=8):
    """
    Creates separate plots for each cluster.

    Args:
        sorted_results: Dictionary mapping cluster labels to sorted image data (images, names, and scores).
        output_dir: Directory to save the plots.
        max_images_per_cluster: Maximum number of images to display per cluster.
    """
    import matplotlib.pyplot as plt

    for label, (sorted_images, _, sorted_ssim_values) in sorted_results.items():
        num_images = min(len(sorted_images), max_images_per_cluster)

        plt.figure(figsize=(num_images * 2, 4))
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(sorted_images[i], cmap='gray')
            plt.axis('off')
            plt.title(f"SSIM: {sorted_ssim_values[i]:.2f}", fontsize=8)

        plt.suptitle(f"Cluster {label}", fontsize=12)
        plt.tight_layout()

        # Save the plot
        output_file = os.path.join(output_dir, f"cluster_{label}.png")
        plt.savefig(output_file)
        plt.close()


def plot_all_clusters(sorted_results, output_dir, max_images_per_cluster=8):
    """
    Plots all clusters and their images in a single plot with cluster names and counts displayed vertically.

    Args:
        sorted_results: Dictionary mapping cluster labels to sorted image data (images, names, and scores).
        output_dir: Directory to save the consolidated plot.
        max_images_per_cluster: Maximum number of images to display per cluster.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import os

    num_clusters = len(sorted_results)

    # Define figure and GridSpec
    fig = plt.figure(figsize=(max_images_per_cluster * 2, num_clusters * 2))
    gs = gridspec.GridSpec(num_clusters, max_images_per_cluster + 1, width_ratios=[0.5] + [1] * max_images_per_cluster)

    for i, (label, (sorted_images, _, sorted_ssim_values)) in enumerate(sorted_results.items()):
        # Get the number of images in the current cluster
        num_images_in_cluster = len(sorted_images)

        # Add vertical cluster name and image count to the first column
        ax_label = fig.add_subplot(gs[i, 0])
        ax_label.axis('off')  # Turn off axis
        ax_label.text(
            0.5, 0.5,
            f"$\\bf{{Cluster\ {label}}}$\n({num_images_in_cluster} images)",  # Bold cluster name
            fontsize=12, ha='center', va='center', rotation=90
        )

        # Add images to subsequent columns
        num_images = min(num_images_in_cluster, max_images_per_cluster)
        for j in range(max_images_per_cluster):
            ax = fig.add_subplot(gs[i, j + 1])
            if j < num_images:
                ax.imshow(sorted_images[j], cmap='gray')  # Display image in grayscale
                ax.set_title(f"SSIM: {sorted_ssim_values[j]:.2f}", fontsize=8)
            else:
                ax.axis('off')  # Turn off empty image slots
            ax.axis('off')

    # Adjust layout for better appearance
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_dir, "all_clusters_with_labels.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Consolidated cluster plot with labels and counts saved to {output_file}")


# -------------------------------------------- MAIN ---------------------------------------------------------
def main(input_images, output_directory, min_clusters=3, max_clusters=10, target_size=(64, 64), cores=8, plots=1, debug_plots=0):
    """Main function to execute image clustering."""
    # Load images and preprocess
    imgs_fn = input_images  # .mrcs file
    ref_ids_fn = imgs_fn.replace('.mrcs', '.txt')
    image_list, image_names = load_and_preprocess_images_from_mrcs(imgs_fn, ref_ids_fn, target_size)

    # Build similarity matrix
    similarity_matrix = build_similarity_matrix(image_list, cpu_numbers=cores)

    # Generate distance matrix
    distance_matrix = generate_distance_matrix(similarity_matrix)

    # PCA to reduce the dimensionality: apply PCA with 95% Variance retention
    vectors, pca_explained_variance_ratio = apply_pca(distance_matrix, variance=0.95)
    # Calculate the cumulative explained variance
    cumulative_variance = pca_explained_variance_ratio.cumsum()
    # Determine the number of components required to explain 95% of the variance
    n_components_95 = next(i for i, cumulative_var in enumerate(cumulative_variance) if cumulative_var >= 0.95) + 1
    print(f"Number of components to explain 95% of the variance: {n_components_95}")

    # Perform t-SNE for visualization of multidimensional clustering into 2D
    tsne_result = apply_tsne_2d(data=vectors)

    if max_clusters == -1:  # Calculate the max number of clusters based on the number of references
        max_clusters = len(image_list) - 2

    image_array = np.array(image_list)
    image_names_array = np.array(image_names)

    # Determine optimal number of clusters
    optimal_clusters = determine_optimal_clusters(vectors, output_directory,
                                                  min_clusters=min_clusters, max_clusters=max_clusters,
                                                  debug_plots=debug_plots)

    optimal_clusters_kmeans = optimal_clusters['kmeans']
    optimal_clusters_hierarchical = optimal_clusters['hierarchical']

    # Comparing Clustering with optimal number of clusters
    # Perform K-means clustering
    kMeans_results = {}
    labels_KMeans = perform_kmeans_clustering(data=vectors, n_clusters=optimal_clusters_kmeans)
    kMeans_results[LABELS] = labels_KMeans

    # Perform Hierarchical clustering
    hierarchical_results = {}
    labels_hierarchical = perform_hierarchical_clustering(data=vectors, n_clusters=optimal_clusters_hierarchical)
    hierarchical_results[LABELS] = labels_hierarchical

    # Compute Silhouette Score for each clustering method
    silhouette_kmeans = silhouette_score(vectors, labels_KMeans)
    silhouette_hierarchical = silhouette_score(vectors, labels_hierarchical)

    print(f'Silhouette Score for K-Means: {silhouette_kmeans}')
    print(f'Silhouette Score for Hierarchical Clustering: {silhouette_hierarchical}')

    kMeans_results[SCORE] = silhouette_kmeans
    hierarchical_results[SCORE] = silhouette_hierarchical

    results = {'kmeans': kMeans_results, 'hierarchical': hierarchical_results}

    # Filter out None results
    valid_results = {method: res for method, res in results.items() if res[SCORE] is not None}

    # Find and print the best method and its Silhouette Score
    if valid_results:
        best_method = max(valid_results, key=lambda method: valid_results[method][SCORE])
        best_result = valid_results[best_method]

        print(f"Best clustering method: {best_method}")
        print(f"Best Silhouette Score: {best_result[SCORE]}")

        filename = os.path.join(output_directory, 'best_results.txt')
        with open(filename, 'w') as file:
            file.write(f"Best clustering method: {best_method}\n")
            file.write(f"Best Silhouette Score: {best_result[SCORE]}")

        # Use the best labels for further analysis or visualization
        best_labels = best_result[LABELS]
    else:
        print("No valid clustering results available.")
        exit(0)

    result_dict = get_images_to_representative_alignment(best_labels, image_array, image_names_array, vectors, cores)
    labels_array, images_array, vectors_array = extract_array_like_results(result_dict)

    # Use t-SNE to have a 2D visual representation of the clustering
    final_tsne_result = apply_tsne_2d(data=vectors_array)

    # Sort images in cluster and write results
    best_clusters_with_names, sorted_results = sort_images_in_cluster(best_labels, result_dict, output_directory)

    # Visualize t-SNE with kmeans clustering labels
    dim = target_size[0]
    zoom = 0.35 if dim > 64 else 0.7

    if plots:
        image_scatter_plot(final_tsne_result, images_array, labels_array, output_directory, zoom=zoom,
                           title='Aligned visualization of best clustered aligned images', cluster_type='best')

        plot_all_clusters(sorted_results, output_directory, max_images_per_cluster=8)
        # In case you want to have one figure per cluster
        # plot_individual_clusters(sorted_results, output_directory, max_images_per_cluster=8)

    if debug_plots:
        # Plot similarity matrix
        plot_similarity_matrix(similarity_matrix,
                               labels=[f'Image {i + 1}' for i in range(len(image_list))],
                               output_directory=output_directory)

        # plot the cumulative explained variance
        plot_PCA(cumulative_variance, output_directory)

        # Visualize t-SNE with kmeans clustering labels
        image_scatter_plot(tsne_result, image_array, labels_KMeans, output_directory,
                           title='t-SNE with K-mean Clustering', cluster_type='kMeans', zoom=zoom)
        # Visualize t-SNE with hierarchical clustering labels
        image_scatter_plot(tsne_result, image_array, labels_hierarchical, output_directory,
                           title='t-SNE with Hierarchical Clustering', cluster_type='hierarchical', zoom=zoom)


class ScriptCl2dClustering(XmippScript):

    def __init__(self):
        XmippScript.__init__(self)

    def defineParams(self):
        self.addUsageLine('Groups similar images into the same class. It use for 2D averages.')
        ## params
        self.addParamsLine(' -i <inputAverages> : .mrcs file containing the 2D averages ' +
                           'images that you want to cluster.\n')

        self.addParamsLine(' -o <outputDir> : output directory where results will be stored.')

        self.addParamsLine('[ -m <minNclusters> ]: (optional) minimum number of clusters. Default 10 clusters.')

        self.addParamsLine(' [ -M <maxNclusters> ]: (optional) maximum number of clusters. Default number of images + 2.')

        self.addParamsLine(' [ -j <cores> ]: (optional) number of cores you want to use for alignment. Default number of cores available found on your machine.')

        ## examples
        self.addExampleLine('xmipp_cl2d_clustering -i path/to/inputAverages.mrcs -o path/to/outputDir -m 10 -M 20 -j 8')
        self.addExampleLine('xmipp_cl2d_clustering -i path/to/inputAverages.mrcs -o path/to/outputDir -m 10 -M -1 -j 8')


    def run(self):
        # Input images
        if self.checkParam('-i'):
            input_images = self.getParam('-i')
            if not os.path.exists(input_images):
                raise Exception('Error, input average .mrcs does not exists.')
        else:
            raise Exception("Error, input average .mrcs is required as argument.")

        # Output dir: the program will create this directory
        if self.checkParam('-o'):
            output_dir = self.getParam('-o')
            create_directory(output_dir)
        else:
            raise Exception("Error, output directory is required as argument.")

        if self.checkParam('-m'):
            min_clusters = self.getIntParam('-m')
        else:
            print("If the minimum number of clusters is not given the default value of 10 is used.")
            min_clusters = 10

        if self.checkParam('-M'):
            max_clusters = self.getIntParam('-M')
        else:
            print("If the maximum number of clusters is not given the default value is used, number of images - 2.")
            max_clusters = 10

        if self.checkParam('-j'):
            cores = self.getIntParam('-j')
        else:
            print("If the number of cores is not given the default value is used, as many cores as they are available.")
            cores = cpu_count()


        main(input_images=input_images, output_directory=output_dir, min_clusters=min_clusters,
             max_clusters=max_clusters, target_size=(64, 64), cores=cores)


if __name__ == '__main__':
    exitCode = ScriptCl2dClustering().tryRun()
    sys.exit(exitCode)