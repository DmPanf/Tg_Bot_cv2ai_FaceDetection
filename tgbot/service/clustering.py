# -*- coding: UTF-8 -*-

# Import libraries
from sklearn.cluster import KMeans #from sklearn import cluster as sk_cluster
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
import io
from scipy.spatial import distance as sci_distance # Определение оптимального количество кластеров методом Elbow
from kneed import KneeLocator # для автоматического определения оптимального числа кластеров
#from sklearn.metrics import silhouette_score # 2-й метод нахождения оптимального числа Кластеров

import matplotlib.image as Img
from scipy.cluster.vq import whiten
from scipy.cluster.vq import kmeans
import pandas as pd
#import matplotlib.pyplot as plt
from skimage import io as IO

#cascadePath = 'tgbot/service/Cascades/..'

# ------> Clustering K-Means <----------
def kmeans_method(in_img, out_img):
    nparr = np.fromstring(in_img.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    mod_img = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA) # Reduce image size to reduce the execution time
    mod_img = mod_img.reshape(mod_img.shape[0]*mod_img.shape[1], 3) # Reduce the input to two dimensions for KMeans
#    print(mod_img.shape)
    N = 15
    sse = {}
    for k in range(2, N):
        clf = KMeans(n_clusters=k, max_iter=100).fit(mod_img)
#        label = clf.labels_
#        print(label)
#        clf = KMeans(n_clusters=k).fit(mod_img)
#        print(k, ' - ', clf, ' = ', clf.inertia_)
        sse[k] = clf.inertia_ # Inertia: Sum of distances of samples to yheir closes cluster center
#    print(sse)
#    kl = KneeLocator(range(1,N), sse, S=0.1, curve='convex', direction='decreasing')
    kn = KneeLocator(x=list(sse.keys()), y=list(sse.values()), curve='convex', direction='decreasing')
    K = kn.knee # + 1
    print('\nОптимальное количество кластеров: ', K)
#    print('\nОптимальное количество кластеров: ', kl.elbow, '\n')
#    kl = silhouette_score(sse, clf.labels_)
#    print('\nОптимальное количество кластеров: ', kl, '\n')

    plt.figure(figsize=(8,5), dpi=100)
#    plt.plot(K, avgWithinSS, 'b*-')
    plt.plot(list(sse.keys()), list(sse.values()), 'b*-', linewidth=2)
    plt.grid(True)
    plt.axvline(x=K, linestyle='--', color='darkred', linewidth=2)
    plt.plot(K, sse[K],'ro') # Выделить точку с оптимальным числом Кластеров
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title(f'Elbow for KMeans clustering (Оптимальное кол-во кластеров: {K})')
    plt.savefig('opt.jpg', bbox_inches='tight')
    plt.close('all')

    new_img = cv2.imread('opt.jpg')
    is_success, buffer = cv2.imencode(".jpg", new_img)
    return io.BytesIO(buffer)

# -------> Color Pie5 <------------
#Define the HEX values of colours
def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def RGB2HEX2(color, percent):
    Color = "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
    Proc = "{:0.0f}%".format(percent * 100)
    txt = Color + " [" + str(Proc) + "]"
    return txt

#Returns the colours in the image
def get_colours(img, no_of_colours, show_chart):
#    img = cv2.imread(img_path)
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the image to original colors i.e. RGB
    mod_img = cv2.resize(img, (600, 400), interpolation = cv2.INTER_AREA) # Reduce image size to reduce the execution time
    mod_img = mod_img.reshape(mod_img.shape[0]*mod_img.shape[1], 3) # Reduce the input to two dimensions for KMeans
    clf = KMeans(n_clusters = no_of_colours)  # Define the clusters
    labels = clf.fit_predict(mod_img)
    counts = Counter(labels)
    counts = dict(sorted(counts.items()))
    Sum = sum(counts.values())
#    print(Sum, counts)
    center_colours = clf.cluster_centers_
    ordered_colours = [center_colours[i] for i in counts.keys()]
    hex_colours = [RGB2HEX(ordered_colours[i]) for i in counts.keys()]
    hex_labels = [RGB2HEX2(ordered_colours[i], counts[i]/Sum) for i in counts.keys()]
    rgb_colours = [ordered_colours[i] for i in counts.keys()]
    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_labels, colors = hex_colours)
        plt.legend(loc='upper left')
        plt.show()
        plt.close('all')
        return
    else:
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_labels, colors = hex_colours)
        plt.legend(loc='upper left')
        plt.savefig('pie.jpg', bbox_inches='tight') # Сохранить Вывод во временный файл без окантовки (молока)
        plt.close('all')
        return rgb_colours

def colorpie_method(in_img, out_img):
    nparr = np.fromstring(in_img.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    N = 5
    #isWritten = cv2.imwrite('in.jpg', image)
    #print(' --------> ')
    #image = cv2.imread('in.jpg')
    rgb_col = get_colours(image, N, False)
#    print(rgb_col)
    new_img = cv2.imread('pie.jpg')
    is_success, buffer = cv2.imencode(".jpg", new_img)
#    is_success, buffer = cv2.imencode(".jpg", 'clust.jpg')
    return io.BytesIO(buffer)


# -----------> Main Bar3 <------------------
def colorbar3_method(in_img, out_img):
    nparr = np.fromstring(in_img.getvalue(), np.uint8)
#    my_image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    my_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#    my_image = Img.imread(img0)
    r = []
    g = []
    b = []
    for row in my_image:
        for temp_r, temp_g, temp_b in row:
            r.append(temp_r)
            g.append(temp_g)
            b.append(temp_b)
    my_df = pd.DataFrame({'red' : r, 'green' : g, 'blue' : b})
    my_df['scaled_color_red'] = whiten(my_df['red'])
    my_df['scaled_color_blue'] = whiten(my_df['blue'])
    my_df['scaled_color_green'] = whiten(my_df['green'])
    #print(my_df)
    cluster_centers, _ = kmeans(my_df[['scaled_color_red', 'scaled_color_blue', 'scaled_color_green']], 3)
    dominant_colors = []
    r_std, g_std, b_std = my_df[['red', 'green', 'blue']].std()
    for cluster_center in cluster_centers:
        r_scal, g_scal, b_scal = cluster_center
        dominant_colors.append((r_scal * r_std / 255, g_scal * g_std / 255, b_scal * b_std / 255))
    #print(dominant_colors)
    plt.figure(figsize=(6,3), dpi=100)
#    plt.imshow([dominant_colors], vmin=0, vmax=255)
    plt.imshow([dominant_colors])
    plt.savefig('bar3.jpg', bbox_inches='tight')
    plt.close('all')

    new_img = cv2.imread('bar3.jpg')
    is_success, buffer = cv2.imencode(".jpg", new_img)
    return io.BytesIO(buffer)

# --------------> Mean Bar7 <-----------
def visualize_colors(cluster, centroids):
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Create frequency rect and iterate through each cluster's color and percentage
    X = 180 # Высота цветной полосы
    Y = 600 # Ширина выводимой палитры
    rect = np.zeros((X, Y, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    start = 0
    for (percent, color) in colors:
        Proc1 = "{:0.2f}%".format(percent * 100)
        Proc2 = "{:0.1f}%".format(percent * 100)
        Color = RGB2HEX(color)
        print(color, Proc1)
        end = start + (percent * Y)
        cv2.rectangle(rect, (int(start), 0), (int(end), X), color.astype("uint8").tolist(), -1)
        cv2.putText(rect, Proc2, (int(start)+2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        cv2.putText(rect, Proc2, (int(start)+3, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        cv2.putText(rect, Color, (int(start)+2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
        start = end
    return rect

def colorbar7_method(in_img, out_img):
    nparr = np.fromstring(in_img.getvalue(), np.uint8)
#    my_image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    N = 7
    reshape = image.reshape((image.shape[0] * image.shape[1], 3))

    # Find and display most dominant colors
    cluster = KMeans(n_clusters=N).fit(reshape)
    visualize = visualize_colors(cluster, cluster.cluster_centers_)
    visualize = cv2.cvtColor(visualize, cv2.COLOR_RGB2BGR)

#    plt.savefig('bar7.jpg', bbox_inches='tight')
#    plt.close('all')

    #new_img = cv2.imread('bar7.jpg')
    is_success, buffer = cv2.imencode(".jpg", visualize)
    return io.BytesIO(buffer)


# ---------> Optimal Bar  <----------
def centroid_histogram(clt):
# grab the number of different clusters and create a histogram based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    hist = hist.astype("float") # normalize the histogram, such that it sums to one
    hist /= hist.sum()
    return hist # return the histogram


def plot_colors(hist, centroids):
    X = 180
    Y = 600
# initialize the bar chart representing the relative frequency of each of the colors
    bar = np.zeros((X, Y, 3), dtype = "uint8")
    startX = 0
# loop over the percentage of each cluster and the color of each cluster
#    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
#	for (percent, color) in zip(hist, centroids):
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    for (percent, color) in colors:
        Color = RGB2HEX(color)
        Proc = "{:0.1f}%".format(percent * 100)
        endX = startX + (percent * Y) # plot the relative percentage of each cluster
        cv2.rectangle(bar, (int(startX), 0), (int(endX), X), color.astype("uint8").tolist(), -1)
        cv2.putText(bar, Proc, (int(startX)+2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        cv2.putText(bar, Proc, (int(startX)+3, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
        cv2.putText(bar, Color, (int(startX)+2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,0), 1)
        startX = endX
    return bar # return the bar chart


def optimal_method(in_img, out_img):
    nparr = np.fromstring(in_img.getvalue(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA) # Reduce image size to reduce the execution time
    image = image.reshape((image.shape[0] * image.shape[1], 3)) # reshape the image to be a list of pixels

    N = 15 # Вычисляем оптимальное число Кластеров
    sse = {}
    for k in range(2, N):
#        clf = KMeans(n_clusters=k, max_iter=100).fit(image)
        clf = KMeans(n_clusters=k).fit(image)
        sse[k] = clf.inertia_ # Inertia: Sum of distances of samples to yheir closes cluster center
    kn = KneeLocator(x=list(sse.keys()), y=list(sse.values()), curve='convex', direction='decreasing')
    K = kn.knee # + 1
    print('\nОптимальное количество кластеров: ', K)
#    N = K # Строим цветовую гамму оптимальным образом
    clt = KMeans(n_clusters=K).fit(image) # cluster the pixel intensities
#    clt.fit(image)
    hist = centroid_histogram(clt) # build a histogram of clusters and then create a figure representing the number of pixels labeled to each color
    bar = plot_colors(hist, clt.cluster_centers_)

    plt.figure(figsize = (8, 6)) # show our color bart
### plt.legend(loc='upper left', labels = hist) # Bad Idea)
    plt.axis("off")
    plt.imshow(bar)

    plt.savefig('opt.jpg', bbox_inches='tight')
    plt.close('all')

    new_img = cv2.imread('opt.jpg')
    is_success, buffer = cv2.imencode(".jpg", new_img)
#    is_success, buffer = cv2.imencode(".jpg", 'clust.jpg')
    return io.BytesIO(buffer)


# ---------> K-Means НЕ РАБОТАЛО <----------
#    N = 15
#    sse = {}
#    X = {}
#    for k in range(1, N+1): # .append(kmeans.inertia_)
#        clf = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10).fit(image)
#        sse[k] = clf.inertia_ # Inertia: Sum of distances of samples to yheir closes cluster center
#        X[k] = k
#        print('k=', k, ' sse[k]=', sse[k])
##        sse.append(clf.inertia_) # Inertia: Sum of distances of samples to yheir closes cluster center
#    kl = KneeLocator(X, sse, S=0.1, curve='convex', direction='decreasing', interp_method='interp1d')
#    print('\nОптимальное количество кластеров: ', kl, '\n')
#

# -----------> НЕ РАБОТАЕТ ПОКА <---------------
# https://translated.turbopages.org/proxy_u/en-ru.ru.57da6141-62990b24-1cbfc611-74722d776562/https/stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
#
def colorbar9_method(in_img, out_img):
    nparr = np.fromstring(in_img.getvalue(), np.uint8)
#    my_image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img0 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    img = IO.imread(img0)[:, :, :-1]
#    img = IO.imread(image)[:, :, :-1]

    average = img.mean(axis=0).mean(axis=0)
    print(average)
    pixels = np.float32(img.reshape(-1, 3))
    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    print(dominant)

    avg_patch = np.ones(shape=img.shape, dtype=np.uint8)*np.uint8(average)
    indices = np.argsort(counts)[::-1]
    freqs = np.cumsum(np.hstack([[0], counts[indices]/float(counts.sum())]))
    rows = np.int_(img.shape[0]*freqs)
    dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
    for i in range(len(rows) - 1):
        dom_patch[rows[i]:rows[i + 1], :, :] += np.uint8(palette[indices[i]])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12,6))
    ax0.imshow(avg_patch)
    ax0.set_title('Average color')
    ax0.axis('off')
    ax1.imshow(dom_patch)
    ax1.set_title('Dominant colors')
    ax1.axis('off')
    plt.show(fig)

    plt.savefig('bar7.jpg', bbox_inches='tight')
    plt.close('all')

    new_img = cv2.imread('bar7.jpg')
    is_success, buffer = cv2.imencode(".jpg", new_img)
    return io.BytesIO(buffer)
