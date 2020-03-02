import cv2 as cv
import edge_detector

# Принимает картинку, возвращает ее же, затем координаты,
def rectangle(image):
    im = cv.imread(image)
    img = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    bin_img = edge_detector.image2template(img)
    contours = edge_detector.template_contour_detector(bin_img)
    contoured_img = edge_detector.contour_apply(img, contours)
    final_img, coords = edge_detector.draw_rect_for_text(im.copy(), contours)
    return coords, final_img, bin_img, contoured_img, contours

# Выводит всю информацию по картинке, используя функцию rectangle
def print_info(img):
    coords, final_img, bin_img, contoured_img, contours = rectangle(img)
    cv.imshow("binary", edge_detector.resize(bin_img.copy()))
    cv.imshow("contoured_img", edge_detector.resize(contoured_img.copy()))
    cv.imshow("final_img", edge_detector.resize(final_img.copy()))
    cv.waitKey(0)

print_info("dir/test1.png")