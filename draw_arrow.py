import cv2

# 读取图片
image = cv2.imread('/aidata/zihao/data/realdata_sampled_20240713/train/task_0230_user_0099_scene_0002_cfg_0001/cam_750612070851/color/1720847837764.png')

# 定义箭头的起点和终点
start_point = (100, 100)
end_point = (300, 300)

# 定义箭头的颜色 (B, G, R) 这里是蓝色
color = (255, 0, 0)

# 定义箭头的粗细
thickness = 2

# 绘制箭头
image_with_arrow = cv2.arrowedLine(image, start_point, end_point, color, thickness)

# 显示带箭头的图片
# cv2.imshow('Image with Arrow', image_with_arrow)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存带箭头的图片
cv2.imwrite('image_with_arrow.jpg', image_with_arrow)
