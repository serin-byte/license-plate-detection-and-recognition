# -- encoding:utf-8 --
import cv2
import numpy as np
import matplotlib.pyplot as plt


class Detect:
    # 找到符合车牌形状的矩形
    def __init__(self, dic_config):
        self.file_path = dic_config['file_path']
        self.save_path = dic_config['save_path']

        # 定义蓝底车牌的hsv颜色区间
        self.lower_blue = np.array([100, 40, 50])
        self.higher_blue = np.array([140, 255, 255])

        self.minPlateRatio = 0.5  # 车牌最小比例
        self.maxPlateRatio = 6  # 车牌最大比例

        self.img = cv2.imread(self.file_path)

    def findPlateNumberRegion(self, img):
        region = []
        # 查找外框轮廓
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # print("contours lenth is :%s" % (len(contours)))
        # 筛选面积小的
        list_rate = []
        for i in range(len(contours)):
            cnt = contours[i]
            # 计算轮廓面积
            area = cv2.contourArea(cnt)
            # 面积小的忽略
            if area < 2000:
                continue
            # 转换成对应的矩形（最小）
            rect = cv2.minAreaRect(cnt)
            # print("rect is:%s" % {rect})
            # 根据矩形转成box类型，并int化
            box = np.int32(cv2.boxPoints(rect))
            # 计算高和宽
            height = abs(box[0][1] - box[2][1])
            width = abs(box[0][0] - box[2][0])
            # 正常情况车牌长高比在2.7-5之间,那种两行的有可能小于2.5，这里不考虑
            ratio = float(width) / float(height)
            rate = self.getxyRate(cnt)
            # print("area", area, "ratio:", ratio, "rate:", rate)
            if ratio > self.maxPlateRatio or ratio < self.minPlateRatio:
                continue
            # 符合条件，加入到轮廓集合
            region.append(box)
            list_rate.append(ratio)
        index = self.getSatifyestBox(list_rate)
        return region[index]

    # 找出最有可能是车牌的位置
    def getSatifyestBox(self, list_rate):
        for index, key in enumerate(list_rate):
            list_rate[index] = abs(key - 3)
        # print(list_rate)
        index = list_rate.index(min(list_rate))
        # print(index)
        return index

    def getxyRate(self, cnt):
        x_height = 0
        y_height = 0
        x_list = []
        y_list = []
        for location_value in cnt:
            location = location_value[0]
            x_list.append(location[0])
            y_list.append(location[1])
        x_height = max(x_list) - min(x_list)
        y_height = max(y_list) - min(y_list)
        return x_height * (1.0) / y_height * (1.0)

    def location(self):

        # 转换成hsv模式图片
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        # cv2.imshow("hsv_img", hsv_img)
        # cv2.waitKey(0)
        # 找到hsv图片下的所有符合蓝底颜色区间的像素点，转换成二值化图像
        mask = cv2.inRange(hsv_img, self.lower_blue, self.higher_blue)
        res = cv2.bitwise_and(self.img, self.img, mask=mask)
        # cv2.imshow("res", res)
        # cv2.waitKey(0)

        # 灰度化
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("gray", gray)
        # cv2.waitKey(0)
        # 高斯模糊：车牌识别中利用高斯模糊将图片平滑化，去除干扰的噪声对后续图像处理的影响
        gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
        # cv2.imshow("gaussian", gaussian)
        # cv2.waitKey(0)

        # sobel算子：车牌定位的核心算法，水平方向上的边缘检测，检测出车牌区域
        sobel = cv2.convertScaleAbs(cv2.Sobel(gaussian, cv2.CV_16S, 1, 0, ksize=3))
        # cv2.imshow("sobel", sobel)
        # cv2.waitKey(0)

        # 进一步对图像进行处理，强化目标区域，弱化背景。
        ret, binary = cv2.threshold(sobel, 150, 255, cv2.THRESH_BINARY)
        # cv2.imshow("binary", binary)
        # cv2.waitKey(0)

        # 进行闭操作，闭操作可以将目标区域连成一个整体，便于后续轮廓的提取
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, element)
        # 进行开操作，去除细小噪点
        # eroded = cv2.erode(closed, None, iterations=1)
        # dilation = cv2.dilate(eroded, None, iterations=1)
        # cv2.imshow("closed", closed)
        # cv2.waitKey(0)

        # 查找并筛选符合条件的矩形区域
        region = self.findPlateNumberRegion(closed)

        # cv2.drawContours(img, [region], 0, (0, 255, 0), 2)

        # cv2.imshow("img", img)

        box = self.findPlateNumberRegion(closed)
        # 返回区域对应的图像
        # 因为不知道，点的顺序，所以对左边点坐标排序
        ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
        xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
        ys_sorted_index = np.argsort(ys)
        xs_sorted_index = np.argsort(xs)

        # 获取x上的坐标
        x1 = box[xs_sorted_index[0], 0]
        x2 = box[xs_sorted_index[3], 0]

        # 获取y上的坐标
        y1 = box[ys_sorted_index[0], 1]
        y2 = box[ys_sorted_index[3], 1]

        # 截取图像
        img_plate = self.img[y1:y2, x1:x2]
        # 保存图像
        cv2.imwrite(self.save_path, img_plate)

        # 调整色差
        b, g, r = cv2.split(self.img)
        origin_img = cv2.merge([r, g, b])

        b1, g1, r1 = cv2.split(img_plate)
        cut_img = cv2.merge([r1, g1, b1])

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(origin_img)
        plt.subplot(2, 1, 2)
        plt.imshow(cut_img)
        plt.show()

        # cv2.imshow('demo', img_plate)
        # print('按任意键退出窗口!')
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return img_plate