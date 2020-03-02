import cv2 as cv
import numpy as np


def image2template(img):
    # Вычисляем среднюю яркость картинки
    aver_brightness = np.mean(img)
    ad_th = aver_brightness + aver_brightness * 0.1
    # Преобразуем картинку в бинарную и используем среднюю яркость(немного выше) как порог
    ret, threshold = cv.threshold(img, ad_th, 255, cv.THRESH_BINARY)
    # Ядро горизонтальное, так лучше накладывается маска
    kernel = np.ones((1,8), np.uint8)
    # Максимально уменьшаем рабочую область и образуем больше границ
    bin_img = cv.morphologyEx(threshold, cv.MORPH_ERODE, kernel, iterations=3)
    # Убираем увеличиваем область обратно, при этом шумов намного меньше
    bin_img = cv.morphologyEx(bin_img, cv.MORPH_DILATE, kernel, iterations=3)
    return bin_img

def template_contour_detector(bin_img):
    # Формируем контуры, аппроксимируем их, аппроксимация небольшая
    contours, _ = cv.findContours(bin_img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    approx = []
    for contour in contours:
        epsilon = 0.01 * cv.arcLength(contour, True)
        approx.append(cv.approxPolyDP(contour, epsilon, True))
    return approx

def contour_apply(img, contours):
    # Для наглядности формируем контуры на картинке
    cv.drawContours(img, contours, -1, (0, 0, 0), 2)
    return img

def draw_rect_for_text(img, contours):
    # Рисуем области по контурам
    (im_w, im_h, _) = img.shape
    rect_coords = []
    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)
        # Отсеиваем очень маленькие области
        if(w > im_w/10 and h > im_h/10):
            img = cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            rect_coords.append([x, y, w, h])
    return img, rect_coords

def resize(img):
    return cv.resize(img, (600, 600))