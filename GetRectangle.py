# -*- coding: utf-8 -*
import cv2
import numpy as np
import cmath as math
from enum import Enum
import glob

root="./cutpic"
class MergeType(Enum):
    left=0
    right=1
    none=2
class Wave:
    wave_start = [0,0]
    wave_top = [0,0]
    wave_end = [0,0]
    def __init__(self, wave_start, wave_top,wave_end):
        self.wave_start = wave_start
        self.wave_top = wave_top
        self.wave_end = wave_end
    def WaveH(self):
        return self.wave_top[0]-self.wave_start[0] if self.wave_start[0]<self.wave_end[0] else self.wave_top[0]-self.wave_end[0]
    def WaveW(self):
        #print(wave_end[1]-wave_start[1])
        return self.wave_end[1]-self.wave_start[1]
class Node(object):
    data=[0,0]

    def __init__(self,val,p=0):
        self.data = val
        self.next = p
    def __repr__(self):
  
        return str(self.data)
class LinkList(object):
    def __init__(self):
        self.head = 0
    def __getitem__(self, key):

        if self.is_empty():
            print ('linklist is empty.')
            return

        elif key <0  or key > self.getlength():
            print ('the given key is error')
            return

        else:
            return self.getitem(key)
    def __setitem__(self, key, value):

        if self.is_empty():
            print ('linklist is empty.')
            return

        elif key <0  or key > self.getlength():
            print ('the given key is error')
            return

        else:
            self.delete(key)
            return self.insert(key)
    def initlist(self,data):
        self.head = Node(data[0])
        p = self.head
        for i in data[1:]:
            node = Node(i)
            p.next = node
            p = p.next
    def getlength(self):

        p =  self.head
        length = 0
        while p!=0:
            length+=1
            p = p.next

        return length
    def is_empty(self):

        if self.getlength() ==0:
            return True
        else:
            return False
    def append(self,item):
        q = Node(item)
        if self.head ==0:
            self.head = q
        else:
            p = self.head
            while p.next:
                p = p.next
            p.next = q
    def appendTree(self,treeHead):
            p = self.head
            while p.next:
                p = p.next
            p.next=treeHead
    def getitem(self,index):
        if self.is_empty():
            print ('Linklist is empty.')
            return
        j = 0
        p = self.head
        while p.next!=0 and j <index:
            p = p.next
            j+=1
        if j ==index:
            return p.data

        else:

            print ('target is not exist!')
    def printf(self):
        if self.head==0:
            print("null tree")
        else:
            p=self.head
            while p:
                print(p.data)
                p=p.next
    def delete_val(self, item):
        if self.head==0:
            return None
        p = self.head
        pre=self.head
        while p.next:
            if p.data==item:
                if pre==self.head:
                    pre=p.next
                else:
                    p=p.next
                return
            else:
                pre=p
                p=p.next

        if p.data==item:
            pre.next=0
    def delete(self,index):
        if self.isEmpty():
            exit(0)
        if index < 0 or index > self.getLength() - 1:
            print
            "\rValue Error! Program Exit."
            exit(0)

        i = 0
        p = self.head
       
        while p.next:
            pre = p
            p = p.next
            i += 1
            if i == index:
                pre.next = p.next
                p = None
                return 1

      
        pre.next = None
    def getItemList(self):
        item_list=[]
        if self.head==0:
            print('Linklist is empty.')
            return item_list
        p=self.head
        while p:
            item_list.append(p.data)
            p=p.next
        return  item_list
    # def WaveW(self):
    #     # print(wave_end[1]-wave_start[1])
    #     return self.wave_end[1] - self.wave_start[1]
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = np.sum(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect
def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    widthB = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    heightB = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
def setTreeTagImage(head,tag_img,val):
    p=head
    while p:
        tag_img[p.data[0]][p.data[1]]=val
        p=p.next
def vertical(img):
    
    cv2.imshow("img",img)
    w, h = img.shape[:2]
    pixdata = img
    outDir="E:/"
    x_array = []
    y_array = []
    startX = 0
    endX = 0
    startY = 0
    endY = 0
    sum=8

#边界提取
    # edges_row = []
    # edges_column = []
    #
    #
    # row_sum=0
    # col_sum=0
    #
    # #边界去黑
    # for i in range(w):  # 逐个判断
    #     for j in range(h):
    #         if pixdata[i][j] == 0:  # 255表示白色
    #             row_sum+=1
    #             # print("纵坐标",j)
    #     if row_sum<(w/2):
    #         edges_row.append(i)
    #         row_sum = 0
    #         break;
    #     row_sum = 0
    # for i in reversed(range(w)):  # 逐个判断
    #     for j in range(h):
    #         if pixdata[i][j] == 0:  # 255表示白色
    #             row_sum+=1
    #     if row_sum<(w/2):
    #         edges_row.append(i)
    #         break;
    #     row_sum=0
    # for j in range(h):  # 逐个判断
    #     for i in range(w):
    #         if pixdata[i][j] == 0:  # 255表示白色
    #             col_sum += 1
    #             # print("横坐标",i)
    #             # print("纵坐标",j)
    #     if col_sum < (h / 2):
    #         edges_column.append(j)
    #         col_sum = 0
    #         break;
    #     col_sum=0
    # for j in reversed(range(h)):  # 逐个判断
    #     for i in range(w):
    #         if pixdata[i][j] == 0:  # 255表示白色
    #             col_sum += 1
    #             # print("横坐标",i)
    #             # print("纵坐标",j)
    #     if col_sum < (h / 2):
    #         edges_column.append(j)
    #         break;
    #     col_sum = 0
    # # 边界去黑
    #
    # #边界提取
    # bottom = min(edges_row)  # 底部
    # print(bottom)
    # top = max(edges_row)  # 顶部
    # print(top)
    # left = min(edges_column)
    # print(left)         #左边界
    # right = max(edges_column)
    # print(right)                #右边界
    # pre2_picture = pixdata[bottom:top,left:right]
    pre2_picture = pixdata
    #cv2.imwrite("F:/emily_65004.png", pre2_picture)
    #w,h=pre2_picture.shape[:2]
    for x in range(w):
        b_count = 0
        for y in range(h):
            if pre2_picture[x, y] == 0: #0黑 有字
                b_count += 1
                break
        if b_count>0:
            if startX == 0 :
                startX = x
        else :
            if startX != 0:
                endX = x
                x_array.append({'startX': startX, 'endX': endX})
                startX = 0
    print(len(x_array))
    for i, item in enumerate(x_array):
        #box = (item['startX'], 0, item['endX'], h)
        GrayImage =pre2_picture[item['startX']:item['endX'],0:h]
        CCLCut(GrayImage,i)
        y_array = []
        #print(1111)
        #cv2.imwrite("E:/" + str(i) + ".png", GrayImage)

        # gh=h
        # gw=item['endX']-item['startX']
        # for y in range(gh):
        #     b_count = 0
        #     for x in range(gw):
        #         if GrayImage[x, y] == 255:   #有字
        #             b_count += 1
        #     if b_count > 0:
        #         if startY == 0:
        #            startY = y
        #            continue
        #     if startY != 0 and b_count<=2:
        #             endY = y
        #             y_array.append({'startY': startY, 'endY': endY})
        #             startY = 0
        # for j,y_item in enumerate(y_array):
        #     cutImage = GrayImage[0:gw, y_item['startY']:y_item['endY']]
        #     sum+=1
        #     cv2.imwrite(outDir + str(i)+"__"+str(j) + ".png",cutImage)
def CCLCut(img,img_num):        #255白
    ls = []
    gray=img
    padding_rate=0.1
    gr_w = gray.shape[0]
    gr_h = gray.shape[1]
    gray = cv2.copyMakeBorder(gray, (int)(gr_w * padding_rate), (int)(gr_w * padding_rate),
                                      (int)(gr_h * padding_rate), (int)(gr_h * padding_rate),
                                      cv2.BORDER_CONSTANT, value=[255, 255, 255])
    #cv2.imshow("ggg",gray)
    img_shape = gray.shape
    w = img_shape[0]
    h = img_shape[1]
    tag_img = np.ones((w, h)).astype(np.int16)
    tag_img = tag_img * -1
    list_lslink = []
    type_num = 0
    # for i in range(w):
    #     for j in range(h):
    #         if img[i][j]==0:
    #             print([i,j])



    for i in range(w):
        if gray[i][0] == 0:  # 0黑有字
            link_list = LinkList()
            tag_img[i][0] = type_num
            # label_set[i][0]=label_set_count
            link_list.append(([i, 0]))
            list_lslink.append(link_list)
            type_num += 1
            # label_set_count+=1
    for j in range(1, h):
        if gray[0][j] == 0:  # 0有字
            link_list = LinkList()
            tag_img[0][j] = type_num
            # label_set[0][j] = label_set_count
            link_list.append(([0, j]))
            list_lslink.append(link_list)
            type_num += 1


    # label_set_count += 1
    count = 0
    for j in range(1, h):
        for i in range(1, w):
            type_list = []
            posi_list = []
            min_index = -1
            if gray[i][j] == 0:
                if gray[i - 1][j] == 0:
                    type_list.append(tag_img[i - 1][j])  # 正上
                    posi_list.append([i - 1, j])
                if gray[i - 1][j - 1] == 0:
                    type_list.append(tag_img[i - 1][j - 1])  # 左上
                    posi_list.append([i - 1, j - 1])
                if gray[i][j - 1] == 0:
                    type_list.append(tag_img[i][j - 1])  # 左前
                    posi_list.append([i, j - 1])
                if i < w - 1:
                    if gray[i + 1][j - 1] == 0:
                        type_list.append(tag_img[i + 1][j - 1])  # 左下
                        posi_list.append([i + 1, j - 1])
                if len(type_list) > 0:
                    min_index = type_list.index(min(type_list))
                    for c in range(len(type_list)):
                        if c != min_index:
                            # print(type_list[min_index])
                            if type_list[min_index] != type_list[c] and list_lslink[type_list[c]] != None:  ##当领域相同时 None一次之后其余变空
                                setTreeTagImage(list_lslink[type_list[c]].head, tag_img, type_list[min_index])
                                list_lslink[type_list[min_index]].appendTree(list_lslink[type_list[c]].head)
                                list_lslink[type_list[c]] = None
                    # 对前链表所有minlabel进行设置
                    tag_img[i][j] = type_list[min_index]
                    list_lslink[type_list[min_index]].append(([i, j]))
                else:
                    link_list = LinkList()
                    tag_img[i][j] = type_num
                    link_list.append([i, j])
                    list_lslink.append(link_list)
                    type_num += 1
    pic = np.zeros((w, h, 3), np.uint8)
    #print(len(list_lslink))

    result_list = []
    for list in list_lslink:
        if list:
            result_list.append(list)
    #print(len(result_list))
    count=0
    for item in result_list:
        count+=1
        left=9999
        right=0
        top=9999
        bottom=0
        p = item.head
        while p:
            if p.data[1]<left:
                left=p.data[1]
            if p.data[1]>right:
                right=p.data[1]
            if p.data[0]<top:
                top=p.data[0]
            if p.data[0]>bottom:
                bottom = p.data[0]
            p=p.next

        item_w=bottom-top+1
        item_h =right-left+1
        result_pic=np.ones((item_w, item_h,1), np.uint8)
        result_pic*=255
        p = item.head

        while p:
            result_pic[p.data[0]-top,p.data[1]-left]=0
            p = p.next
        #result_pic = img[bottom:top,left:right]

        # padding_rate=0.1
        # gr_w = result_pic.shape[0]
        # gr_h = result_pic.shape[1]
        # constant = cv2.copyMakeBorder(result_pic, (int)(gr_w * padding_rate), (int)(gr_w * padding_rate),
        #                               (int)(gr_h * padding_rate), (int)(gr_h * padding_rate),
        #                               cv2.BORDER_CONSTANT, value=[255, 255, 255])
        # cv2.imshow("constant", constant)

        AdhereCut(result_pic,str(img_num)+"."+str(count))
        #result_pic=cv2.resize(result_pic, (28, 28), interpolation=cv2.INTER_LINEAR)
       # cv2.imshow("pic", result_pic)
        #cv2.imwrite("E:/" + str(count) + ".png", result_pic)
        #result_pic= np.zeros((right-left, top-bottom, 3), np.uint8)
    
    
    
    # for item in result_list:
    #     #     pic[item[0]][item[1]]=(0,0,255)
    #     p = item.head
    #     while p:
    #         pic[p.data[0]][p.data[1]] = (0, 0, 255)
    #         p = p.next
def  AdhereCut(img,img_num):

    # padding_rate=0.2
    # gr_w = img.shape[0]
    # gr_h = img.shape[1]
    # constant = cv2.copyMakeBorder(img, (int)(gr_w * padding_rate), (int)(gr_w * padding_rate),
    #                               (int)(gr_h * padding_rate), (int)(gr_h * padding_rate),
    #                               cv2.BORDER_CONSTANT, value=[255, 255, 255])
    # cv2.imshow("constant", constant)


    img_shape = img.shape
    ls = []
    w = img_shape[0]
    h = img_shape[1]
    gray=img
    #cv2.imshow("gray",gray)
    #cv2.waitKey(0)

    # 二值化处理
    #_, gray = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
    pic = np.zeros((w, h, 3), np.uint8)
    # 画出下轮廓
    for y in range(0, h):
        lowPoint = [0, 0]
        exist = 0
        for x in range(0, w):
            if gray[x, y] == 0:  # 0黑色
                if x > lowPoint[0]:
                    exist = 1
                    pic[lowPoint[0], lowPoint[1]] = (255, 255, 255)
                    lowPoint = [x, y]
                else:
                    pic[x, y] = (255, 255, 255)
                # out[lowPoint[0],lowPoint[1]]=255
            else:
                pic[x, y] = (255, 255, 255)
        if exist:
            ls.append(lowPoint)
            pic[lowPoint[0], lowPoint[1]] = (0, 0, 255)



    if len(ls)==0:
        return


    #cv2.imshow('origi', pic)


    # 得到下轮廓波峰和波谷
    wave_list = []
    num = 0
    wave_start = ls[0]
    wave_start_h = ls[0][0]    #//x 坐标值
    wave_top_h = wave_start_h
    wave_top = []   #top坐标值
    wave_end_h = 0
    wave_end = []   #end坐标值
    top_find = 0
    for i in range(0, len(ls)):
        if top_find == 1:
            if ls[i][0] <= wave_end_h:
                wave_end_h = ls[i][0]
                wave_end = ls[i]
            else:
                num += 1
                wave_list.append(Wave(wave_start, wave_top, wave_end))
                wave_start = wave_end
                wave_start_h = wave_end_h
                wave_top = []
                wave_end = []
                top_find = 0
                wave_top_h = wave_start_h
            if i == len(ls) - 1:
                num += 1
                wave_list.append(Wave(wave_start, wave_top, wave_end))
        else:
            if ls[i][0] >= wave_top_h:
                wave_top = ls[i]
                wave_top_h = ls[i][0]
                if i == len(ls) - 1:
                    num += 1
                    wave_list.append(Wave(wave_start, wave_top, wave_top))  #一条直线时
            else:
                top_find = 1
                wave_end_h = ls[i][0]
                wave_end = ls[i]
                if i == len(ls) - 1:
                    num += 1
                    wave_list.append(Wave(wave_start, wave_top, wave_end))
    avg_h = 0
    avg_w = 0

    if num==0:
        return

    for i in wave_list:
        avg_h += i.WaveH()
        avg_w += i.WaveW()
        print("waveH:" + str(i.WaveH()))
    avg_h /= num
    avg_w /= num

    wave_list_result = []

    list_length = len(wave_list)

    # 波峰筛选
    wave_list_result = []
    count = 1
    mergeType = MergeType.none

    merge_wave = wave_list[0]
    while count < list_length:
        # new_wave=Wave()
        if merge_wave.WaveH() < avg_h * 0.1:
            if len(wave_list_result) > 0:
                if count == list_length - 1:
                    mergeType = MergeType.left
                else:
                    last_wave = wave_list_result[len(wave_list_result) - 1]
                    r_wave = wave_list[count]
                    if last_wave.WaveH() < r_wave.WaveH():
                        mergeType = MergeType.left
                    else:
                        mergeType = MergeType.right
            else:
                mergeType = MergeType.right
        elif merge_wave.WaveH() < avg_w * 0.3:
            if len(wave_list_result) > 0:
                if count == list_length - 1:
                    mergeType = MergeType.left
                else:
                    last_wave = wave_list_result[len(wave_list_result) - 1]
                    r_wave = wave_list[count]
                    if last_wave.WaveW() < r_wave.WaveW():
                        mergeType = MergeType.left
                    else:
                        mergeType = MergeType.right
            else:
                mergeType = MergeType.right
        elif abs(merge_wave.WaveH() - wave_list[count].WaveH()) > 0.80 * max(merge_wave.WaveH(),
                                                                             wave_list[count].WaveH()):
            mergeType = MergeType.right

        else:
            print("none")
            wave_list_result.append(merge_wave)
            merge_wave = wave_list[count]
            count += 1
            mergeType = MergeType.none
        if mergeType == MergeType.left:
            # if len(wave_list_result)>0:
            index = len(wave_list_result) - 1
            last_wave = wave_list_result[index]
            l_top = last_wave.WaveH()
            m_top = merge_wave.WaveH()
            last_wave.wave_top = last_wave.wave_top if l_top > r_top else merge_wave.wave_top
            last_wave.wave_end = merge_wave.wave_end
            count += 1
            print("left")
        elif mergeType == MergeType.right:
            m_top = merge_wave.WaveH()
            r_top = wave_list[count ].WaveH()
            merge_wave.wave_end = wave_list[count ].wave_end
            merge_wave.wave_top = merge_wave.wave_top if m_top > r_top else wave_list[count].wave_top
            count += 1
            print("right")

    wave_list_result.append(merge_wave)

#合并后的波如果只有1 直接存退出
    if len(wave_list_result)<=1:

        padding_rate=0.2
        # gr_w = img.shape[0]
        # gr_h = img.shape[1]
        # constant = cv2.copyMakeBorder(img, (int)(gr_w * padding_rate), (int)(gr_w * padding_rate),
        #                               (int)(gr_h * padding_rate), (int)(gr_h * padding_rate),
        #                               cv2.BORDER_CONSTANT, value=[255, 255, 255])
        #
        # cv2.imshow("constant", constant)

        gr_w=gray.shape[0]
        gr_h=gray.shape[1]
        constant=[]
        if gr_w>gr_h:
            diff_value = (gr_w-gr_h)*(1+padding_rate*2)
            constant = cv2.copyMakeBorder(gray,(int)(gr_w * padding_rate), (int)(gr_w * padding_rate),
                                          (int)(diff_value/2+gr_h * padding_rate), int(diff_value/2+gr_h * padding_rate), cv2.BORDER_CONSTANT,
                                          value=[255, 255, 255])
        else :
            diff_value = (gr_h - gr_w)*(1+padding_rate*2)
            constant = cv2.copyMakeBorder(gray,  (int)(diff_value/2+gr_w * padding_rate), int(diff_value/2+gr_w * padding_rate),
                                          (int)(gr_h * padding_rate), (int)(gr_h * padding_rate),cv2.BORDER_CONSTANT,
                                          value=[255, 255, 255])
        print(constant.shape)
        #cv2.imshow("cons",constant)
        result_pic = cv2.resize(constant, (28, 28), interpolation=cv2.INTER_LINEAR)
        imgFix = np.zeros((28, 28, 1), np.uint8)
        for i in range(28):
            for j in range(28):
                imgFix[i, j] = 255 - result_pic[i, j]
       # result_pic_gray = cv2.cvtColor(result_pic, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(root + img_num + ".png", imgFix)
        return
    count = 0
    cut_posi = []
    cut_left = wave_list_result[0].wave_start[1]
    cut_right = wave_list_result[0].wave_start[1]
    while count < len(wave_list_result) - 1:
        cut_left = cut_right
        l_wave = wave_list_result[count]
        r_wave = wave_list_result[count + 1]
        start_posi = [0, 0]
        if l_wave.wave_top[0] > r_wave.wave_top[0]:
            start_posi = r_wave.wave_top

            find_black = 0
            for y in reversed(range(r_wave.wave_top[1] + 1)):
                if find_black == 1:
                    if (gray[start_posi[0]][y]) == 0:
                        start_posi = [start_posi[0], y + 1]
                        break
                else:
                    if (gray[start_posi[0]][y]) == 255:
                        find_black = 1
                        start_posi = [start_posi[0], y]
        else:
            start_posi = l_wave.wave_top
            y = start_posi[1]
            while y <= h:
                if (gray[start_posi[0]][y]) == 0:  # 0
                    start_posi = [start_posi[0], y]
                    break
                y += 1
        refer_Y = wave_list_result[count + 1].wave_start[1]
        start_x = start_posi[0]
        start_y = start_posi[1]
        print("start_y" + str(start_y))
        print("refer_Y" + str(refer_Y))
        while start_x > 0 and start_y > 0 and start_y < h:

            if start_y < refer_Y:  # 右优先
                if gray[start_x - 1][start_y + 1] == 255:  # 255 右上
                    start_x = start_x - 1
                    start_y = start_y + 1
                elif gray[start_x - 1][start_y] == 255:  # 255 上
                    start_x = start_x - 1
                elif gray[start_x - 1][start_y - 1] == 255:  # 255 左上
                    start_x = start_x - 1
                    start_y = start_y - 1
                elif gray[start_x][start_y + 1] == 255:  # 255 右
                    start_y = start_y + 1
                elif gray[start_x][start_y - 1] == 255:  # 255 左
                    start_y = start_y + 1
                else:
                    start_x = start_x - 1
            elif  start_y> refer_Y:  # 左优先
                if gray[start_x - 1][start_y - 1] == 255:  # 255 左上
                    start_x = start_x - 1
                    start_y = start_y - 1
                elif gray[start_x - 1][start_y] == 255:  # 255 下
                    start_x = start_x - 1
                elif gray[start_x - 1][start_y + 1] == 255:  # 255 右上
                    start_x = start_x - 1
                    start_y = start_y + 1
                elif gray[start_x][start_y - 1] == 255:  # 255 左
                    start_y = start_y - 1
                elif gray[start_x][start_y + 1] == 255:  # 255 右
                    # if start_y-1==refer_Y:
                    #     start_y=refer_Y
                    #     break
                    start_y = start_y - 1
                else:
                    start_x = start_x - 1
            if start_y == refer_Y: break
        end_posi = [start_x, start_y]
        cut_right = start_y
        print("end_posi" + str(end_posi))
        cut = gray[0:w, cut_left:cut_right]
        gr_w = cut.shape[0]
        gr_h = cut.shape[1]

        #cv2.imshow("cut", cut)
        #cv2.waitKey(0)

        top_posi=0
        bottom_posi = w
        find=0
        for i in range(gr_w):
            for j in range(gr_h):
                if cut[i][j]==0:
                    top_posi=i
                    find=1
                    break
            if find:
                break
        find=0
        for i in reversed(range(gr_w)):
            for j in range(gr_h):
                if cut[i][j] == 0:
                    bottom_posi = i
                    find = 1
                    break
            if find:
                break
        print("top ,bottom")
        print(top_posi,bottom_posi)
        cut = cut[top_posi:bottom_posi, 0: gr_h]

        #cv2.imshow("cut",cut)
        #cv2.waitKey(0)
        constant = []
        padding_rate = 0.2
        gr_w = cut.shape[0]
        gr_h = cut.shape[1]
        constant = []
        if gr_w > gr_h:
            diff_value = (gr_w - gr_h) * (1 + padding_rate * 2)
            constant = cv2.copyMakeBorder(cut, (int)(gr_w * padding_rate), (int)(gr_w * padding_rate),
                                          (int)(diff_value / 2 + gr_h * padding_rate),
                                          int(diff_value / 2 + gr_h * padding_rate), cv2.BORDER_CONSTANT,
                                          value=[255, 255, 255])
        else:
            diff_value = (gr_h - gr_w) * (1 + padding_rate * 2)
            constant = cv2.copyMakeBorder(cut, (int)(diff_value / 2 + gr_w * padding_rate),
                                          int(diff_value / 2 + gr_w * padding_rate),
                                          (int)(gr_h * padding_rate), (int)(gr_h * padding_rate), cv2.BORDER_CONSTANT,
                                          value=[255, 255, 255])



        # gr_w = cut.shape[0]
        # gr_h = cut.shape[1]
        # if gr_w > gr_h:
        #     diff_value = gr_w - gr_h
        #     constant = cv2.copyMakeBorder(cut, 0, 0,
        #                                   (int)(diff_value / 2), int(diff_value / 2), cv2.BORDER_CONSTANT,
        #                                   value=[255, 255, 255])
        # else:
        #     diff_value = gr_h - gr_w
        #     constant = cv2.copyMakeBorder(cut, (int)(diff_value/2), int(diff_value/2),
        #                                   0, 0, cv2.BORDER_CONSTANT,
        #                                   value=[255, 255, 255])
        print(constant.shape)
        result_pic = cv2.resize(constant, (28, 28), interpolation=cv2.INTER_LINEAR)
        imgFix = np.zeros((28,28, 1), np.uint8)
        for i in range(28):
            for j in range(28):
                imgFix[i, j] = 255 - result_pic[i, j]
        cv2.imwrite(root +img_num+"."+str(count) +".png", imgFix)
        # for i in range(w):
        #     gray[i][start_y] = 0 //分界线上色
        count += 1
    cut = gray[0:w, cut_right:wave_list_result[count].wave_end[1]]
    gr_w = cut.shape[0]
    gr_h = cut.shape[1]
    top_posi = 0
    bottom_posi = w
    find = 0
    for i in range(gr_w):
        for j in range(gr_h):
            if cut[i][j] == 0:
                top_posi = i
                find = 1
                break
        if find:
            break
    find = 0
    for i in reversed(range(gr_w)):
        for j in range(gr_h):
            if cut[i][j] == 0:
                bottom_posi = i
                find = 1
                break
        if find:
            break

    cut = cut[top_posi:bottom_posi, 0: gr_h]

    constant = []
    padding_rate=0.2
    gr_w = cut.shape[0]
    gr_h = cut.shape[1]
    constant = []
    if gr_w > gr_h:
        diff_value = (gr_w - gr_h) * (1 + padding_rate * 2)
        constant = cv2.copyMakeBorder(cut, (int)(gr_w * padding_rate), (int)(gr_w * padding_rate),
                                      (int)(diff_value / 2 + gr_h * padding_rate),
                                      int(diff_value / 2 + gr_h * padding_rate), cv2.BORDER_CONSTANT,
                                      value=[255, 255, 255])
    else:
        diff_value = (gr_h - gr_w) * (1 + padding_rate * 2)
        constant = cv2.copyMakeBorder(cut, (int)(diff_value / 2 + gr_w * padding_rate),
                                      int(diff_value / 2 + gr_w * padding_rate),
                                      (int)(gr_h * padding_rate), (int)(gr_h * padding_rate), cv2.BORDER_CONSTANT,
                                      value=[255, 255, 255])

    # if gr_w > gr_h:
    #     diff_value = gr_w - gr_h
    #     constant = cv2.copyMakeBorder(cut, 0, 0,
    #                                   (int)(diff_value / 2), int(diff_value / 2), cv2.BORDER_CONSTANT,
    #                                   value=[255, 255, 255])
    # else:
    #     diff_value = gr_h - gr_w
    #     constant = cv2.copyMakeBorder(cut, (int)(diff_value/2), int(diff_value/2),
    #                                   0, 0, cv2.BORDER_CONSTANT,
    #                                   value=[255, 255, 255])
    print(constant.shape)
    result_pic = cv2.resize(constant, (28, 28), interpolation=cv2.INTER_LINEAR)
    imgFix = np.zeros((28, 28, 1), np.uint8)
    for i in range(28):
        for j in range(28):
            imgFix[i, j] = 255 - result_pic[i, j]
    #result_pic_gray = cv2.cvtColor(result_pic, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("pic", result_pic)
    cv2.imwrite(root + img_num+"."+str(count) + ".png", imgFix)
    #cv2.imshow('cut', gray)



original_img = cv2.imread(root+"../"+"IMG4.PNG") #F:/handwriting.png
#bg=cv2.imread(root+"IMG_BG.png")
img_shape= original_img.shape

w=img_shape[0]
h=img_shape[1]

maxArea=-9999
maxindex=0
#灰度图
gray=cv2.cvtColor(original_img,cv2.COLOR_BGR2GRAY)


#二值化处理
#cv2.threshold（gray, threshold, if(>threshold)= max）
#返回值： threshold,img

#retval,
im_fixed=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C ,cv2.THRESH_BINARY,27,27)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))

#
erosion = cv2.erode(im_fixed,kernel,iterations = 1)
vertical(erosion)



#识别
deploy=root + '/lenet_deploy.prototxt'    #deploy文件
caffe_model=root + '/lenet_solver_iter_10000.caffemodel'   #训练好的 caffemodel

img_list_paths=glob.glob(r""+root+"/*.png")

print(img_list_paths)

img_list_paths.sort(key=lambda x:tuple(int(v) for v in x.replace(root+"/", '').replace(".png", '').split(".")))


#img=root+'/2__7.png'    #随机找的一张待测图片
labels_filename = root + '/labels.txt'  #类别名称文件，将数字标签转换回类别名称

net = caffe.Net(deploy,caffe_model,caffe.TEST)   #加载model和network
#图片预处理设置
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  #设定图片的shape格式(1,3,28,28)
transformer.set_transpose('data', (2,0,1))    #改变维度的顺序，由原始图片(28,28,3)变为(3,28,28)
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))    #减去均值，前面训练模型时没有减均值，这儿就不用
transformer.set_raw_scale('data', 255)    # 缩放到【0，255】之间
#transformer.set_channel_swap('data', (2,1,0))   #交换通道，将图片由RGB变为BGR

#im=caffe.io.load_image(img)                   #加载图片
result=""
for i in range(len(img_list_paths)):
    img=img_list_paths[i]
    im=caffe.io.load_image(img,False)
    #cv2.imshow("111",im)
    net.blobs['data'].data[...] = transformer.preprocess('data',im)      #执行上面设置的图片预处理操作，并将图片载入到blob中

    #执行测试
    out = net.forward()

    labels = np.loadtxt(labels_filename, str, delimiter='\t')   #读取类别名称文件
    prob= net.blobs['prob'].data[0].flatten() #取出最后一层（Softmax）属于某个类别的概率值，并打印
    #print (prob)
    order=prob.argsort()[-1]  #将概率值排序，取出最大值所在的序号
    result+=labels[order]+" "
    #print ('the class is:',labels[order])  #将该序号转换成对应的类别名称，并打印

print(result)

