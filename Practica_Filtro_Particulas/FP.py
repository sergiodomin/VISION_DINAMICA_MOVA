import glob
import cv2
import numpy as np
import argparse
import os
from random import randint, uniform, gauss

lx = 10
ly = 10
num_particles = 100
pc_base = 0.1


def substraction_images(pre, post, file_sub):
    """
    Restamos la imagen actual con la una imagen fruto de la media de todas las imagenes anteriores

    :param pre: imagen actual
    :param post: media de imagenes anteriores
    :return: imagen restada sin fondo
    """

    x, y, c = images[0].shape
    th = 20
    static = abs(pre - post) > th
    static = static[:, :, 0] * static[:, :, 1] * static[:, :, 2]

    mask = (np.zeros((x, y)) + static)*255

    erode = cv2.erode(mask, None, iterations=2)
    dilate = cv2.dilate(erode, None, iterations=6)
    cv2.imwrite(FLAGS.path_out + '/imgSub/' + file_sub, dilate)

    return dilate


def mean_images_rgb(images_mean):
    """
    Media de las imagenes de entrada

    :param images_mean: array de imagenes
    :return: imagen con la media de las imagenes de entrada
    """
    x, y, c = images_mean[0].shape
    image_b = images_mean[:, :, :, 0]
    image_g = images_mean[:, :, :, 1]
    image_r = images_mean[:, :, :, 2]
    mean_img = np.zeros((x, y, 3))
    mean_img[:, :, 0] = (np.mean(image_b, axis=0))
    mean_img[:, :, 1] = (np.mean(image_g, axis=0))
    mean_img[:, :, 2] = (np.mean(image_r, axis=0))
    return mean_img


def initialization(img_init):
    """
    Se inicializan las coordenadas sobre el ancho y largo del espacio de la imagen

    :param img_init: imagen inicial de entrada
    :return: coordenadas aleatorias sobre el ancho y largo del espacio de la imagen
    """
    y, x, c = img_init.shape
    x_rand = [randint(lx, x-lx) for p in range(0, num_particles)]
    y_rand = [randint(lx, y-lx) for p in range(0, num_particles)]

    points = []
    for i in range(0, len(x_rand)):
        points.append((int(x_rand[i] - lx), int(y_rand[i] - ly), lx, ly, 0, 0))
    return points


def evaluation(points_init_in, img):
    """
    Se calculan si las coordenadas estan sobre la mascara del objeto y superan un cierto solapamiento

    :param points_init_in: coordenadas aleatorias iniciales
    :param img: imagen de entrada, mascara del objeto
    :return: coordenadas de las particulas que han sobrevivido y sus pesos
    """

    # Paara calcular si una particula es buena, calculo cuanto porcentaje de su propia area tiene pixeles blancos
    # pienso que de esta forma, el calculo es mas invariante al tamano de la particula
    weight_particles = []
    points_eval = []
    lx = 0
    ly = 0
    for p in points_init_in:
        x = p[0]
        y = p[1]
        if p[2] != 0:
            lx = p[2]
        if p[3] != 0:
            ly = p[3]
        white_px_obj = ((2 * lx) * (2 * ly)) ** 2
        crop_img = img[y-ly:y+ly, x-lx:x+lx]
        white_px_part = np.sum(crop_img)
        if white_px_part > 0:
            pc = white_px_part / white_px_obj
            if pc > pc_base:
                points_eval.append((x, y, p[2], p[3], p[4], p[5]))
                weight_particles.append(pc)
    return weight_particles, points_eval


def estimation(points_eval_in, img_estimation, file_estimation):
    """
    Se hace un promedio de las particulas que han sobrevivido y se proyecta sobre la imagen

    :param points_eval_in: coordenadas de las particulas
    :param img_estimation: imagen de entrada
    :param file_estimation: fichero de salida
    :return: coordenadas de las nuevas particulas con el tamano estimado de la particula que engloba al objeto
    """
    img_rect = img_estimation.copy()
    arr_x = []
    arr_y = []
    points_estimation = []
    dist_x = 0
    dist_y = 0
    for p in points_eval_in:
        x = p[0]
        y = p[1]

        if p[2] != 0:
            lx = p[2]
        if p[3] != 0:
            ly = p[3]

        arr_x.append(x)
        arr_y.append(y)

    if len(points_eval_in) > 0:
        max_x = max(arr_x)
        max_y = max(arr_y)

        min_x = min(arr_x)
        min_y = min(arr_y)

        dist_x = int((max_x - min_x)/2)
        dist_y = int((max_y - min_y)/2)

        upper_left = (min_x - int(lx/2)), (min_y - int(ly/2))

        bottom_right = (max_x + int(lx/2)), (max_y + int(ly/2))

        img_rect = cv2.rectangle(img_rect, upper_left, bottom_right, (0, 255, 0), 1)

        cv2.imwrite(FLAGS.path_out + '/img/' + file_estimation, img_rect)

    else:
        cv2.imwrite(FLAGS.path_out + '/img/' + file_estimation, img_estimation)

    for p in points_eval_in:
        points_estimation.append((p[0], p[1], dist_x, dist_y, p[4], p[5]))

    return points_estimation


def selection(points_estimation_in, weights):
    """
    Mediante el mecanismo de la ruleta se generan nuevas coordenadas para crear nuevas particuls

    :param points_estimation_in: coordenadas de entrada que han sobrevivido
    :param weights: peso de las coordenadas de entrada
    :return: coordenadas de las nuevas particulas
    """

    sum_weights = np.sum(weights)
    items = []
    points_selection = []
    if np.sum(sum_weights) > 0:
        # print(len(points_estimation_in))
        # print(len(weights))
        for i in range(0, num_particles):
            pick = uniform(0, sum_weights)
            current = 0
            for w in weights:
                current += w
                if current > pick:
                    item = np.where(weights == w)[0][0]
                    items.append(item)
                    break
    for item in items:
        points_selection.append(points_estimation_in[item])
    return points_selection


def diffusion(points_diffusion_in):
    """
    Tras haber generado nuevas particulas estas se pasan por un mecanismo de difusion para repartirlas sutilmente

    :param points_diffusion_in: coordenadas de las particulas
    :return: coordenadas ligeramente repartidas
    """
    points_difussion = []

    for p in points_diffusion_in:
        point = []
        x_rand = gauss(0, 10)
        y_rand = gauss(0, 10)
        x = int(p[0] + x_rand)
        y = int(p[1] + y_rand)
        point.append(x)
        point.append(y)
        points_difussion.append((x, y, p[2], p[3], p[4], p[5]))
    return points_difussion


def motion_model(points_motion_in):
    """
    Calculamos el modelo de movimiento a partir de un modelo regresivo de primer orden, donde aplicamos una gaussiana
    al la velocidad inicial
    :param points_motion_in: coordenadas de difusion de entrada
    :return: coordenadas con modelado de movimiento y velocidad en el eje x y en el eje y
    """
    points_model = []
    arr_vx = []
    arr_vy = []
    for p in points_motion_in:
        vx_rand = gauss(0, 5)
        vy_rand = gauss(0, 5)
        # vx(t+1) = vx(t) + G(0,s)
        vx = int(p[4] + vx_rand)
        # vy(t+1) = vy(t) + G(0,s)
        vy = int(p[5] + vy_rand)
        # x(t+1) = x(t) + vx(t+1)
        x = p[0] + vx
        # y(t+1) = y(t) + vy(t+1)
        y = p[1] + vy
        points_model.append((x, y, p[2], p[3], vx, vy))
        arr_vx.append(vx)
        arr_vy.append(vy)

    return points_model, np.mean(arr_vx), np.mean(arr_vy)


def print_points(img_print, points_print, mean_vx, mean_vy, file_print):
    img_out = img_print.copy()
    for p in points_print:
        x = p[0]
        y = p[1]
        lx = 0
        ly = 0
        if p[2] != 0:
            lx = p[2]
        if p[3] != 0:
            ly = p[3]
        upper_left = (x - lx), (y - ly)

        bottom_right = (x + lx), (y + ly)

        img_out = cv2.rectangle(img_out, upper_left, bottom_right, (0, 255, 0), 1)
        img_out =cv2.putText(img_out, text="vel x: "+ str(mean_vx) , org=(0,25),fontFace=0, fontScale=0.5, color=(0, 255, 0))
        img_out =cv2.putText(img_out, text="vel y: "+ str(mean_vy) , org=(0,50),fontFace=0, fontScale=0.5, color=(0, 255, 0))

    cv2.imwrite(FLAGS.path_out + '/imgVel/' + file_print, img_out)


def make_video(folder, name):
    import cv2
    import numpy as np
    import glob

    img_array = []
    for filename in glob.glob(folder + '*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)

    out = cv2.VideoWriter(FLAGS.path_out + '/video/' + name + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == '__main__':  # pragma: no cover
    ap = argparse.ArgumentParser(description='Main parser')
    ############################################################################
    #   Rutas a modificar en caso de que no se ejecute por línea de comandos   #
    #           Leer fichero leeme.txt para mayor informacion                 #
    ap.add_argument('--path_im', default='D:/Google Drive/MOVA/2_Cuatri/Dinamica/Practicas/P2/trunk/Practica_Filtro_Particulas/SecuenciaPelota')
    ap.add_argument('--path_out', default='D:/Google Drive/MOVA/2_Cuatri/Dinamica/Practicas/P2/trunk/Practica_Filtro_Particulas/out')
    #                                                                          #
    #                                                                          #
    ############################################################################
    FLAGS = ap.parse_args()

    if not os.path.exists(FLAGS.path_out):
        os.makedirs(FLAGS.path_out)
    if not os.path.exists(FLAGS.path_out + '/imgVel'):
        os.makedirs(FLAGS.path_out + '/imgVel')
    if not os.path.exists(FLAGS.path_out + '/img'):
        os.makedirs(FLAGS.path_out + '/img')
    if not os.path.exists(FLAGS.path_out + '/imgSub'):
        os.makedirs(FLAGS.path_out + '/imgSub')
    if not os.path.exists(FLAGS.path_out + '/video'):
        os.makedirs(FLAGS.path_out + '/video')

    num_img_base = 40
    images = []
    first_image = True

    for filename in glob.glob(os.path.join(FLAGS.path_im, '*')):
        file = filename.split('/')[-1].split('\\')[-1]
        img = cv2.imread(filename)
        if first_image:
            points_init = initialization(img)
            first_image = False
        images.append(img)
        images_np = np.array(images)
        n_img = len(images)
        if n_img > num_img_base:
            img_mask = substraction_images(img, mean_images_rgb(images_np[-num_img_base:]), file)
        else:
            img_mask = substraction_images(img, mean_images_rgb(images_np), file)
        weight_particles_out, points_eval_out = evaluation(points_init, img_mask)
        if not points_eval_out:
            points_init = initialization(img)
            weight_particles_out, points_eval_out = evaluation(points_init, img_mask)
        points_estimation_out = estimation(points_eval_out, img, file)
        points_selection_out = selection(points_estimation_out, weight_particles_out)
        points_diffusion = diffusion(points_selection_out)
        points_init, mean_vx, mean_vy = motion_model(points_diffusion)
        print_points(img, points_init, mean_vx, mean_vy, file)

    make_video(FLAGS.path_out + '/img/', 'videoEstimation')
    make_video(FLAGS.path_out + '/imgVel/', 'videoVel')

