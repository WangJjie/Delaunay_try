import numpy as np
import matplotlib.pyplot as plt
import math
from itertools import combinations
from scipy.spatial import Delaunay
import time


points = np.random.randn(40,2)
#print("排序前的坐标",'\n',points)
index_sorted = np.argsort(points[:,0])
points_sorted = points[index_sorted]
#print("排序后的坐标",'\n',points_sorted)


"""定义超级三角形顶点的坐标"""
x_mean = np.mean(points_sorted[:,0])
y_mean = np.mean(points_sorted[:,1])
x_max = np.max(points_sorted[:,0])
y_max = np.max(points_sorted[:,1])
x_min = np.min(points_sorted[:,0])
y_min = np.min(points_sorted[:,1])
print("各点x，y的基础数据信息",'\n',x_max, y_max, x_min, y_min)


#规定超级三角形的内切圆的圆心坐标以及内切圆半径
xc = x_min + (x_max-x_min)/2
yc = y_min + (y_max-y_min)/2
r = max(abs((x_max-x_min)/2), abs((y_max-y_min)/2)) + (x_max-x_min) #保证囊括所有的点，并且点不在超级三角形的边上

#求超级三角形的三点坐标
def calc_incenter(xc, yc, r):
    R = r / math.tan(math.pi/6)
    x1, y1 = xc - R, yc - r
    x2, y2 = xc, yc + r / math.sin(math.pi/6)
    x3, y3 = xc + R, yc - r
    return np.array([x1, y1]), np.array([x2, y2]), np.array([x3, y3])

super_vertice = np.zeros([3,2])
A,B,C = calc_incenter(xc, yc, r)
print(A,B,C)
super_vertice[0,:] = A
super_vertice[1,:] = B
super_vertice[2,:] = C
print(super_vertice)


"""将super_vertice与生成点坐标整合到一个数组里"""
final_array = np.insert(points_sorted,0,super_vertice,axis=0)
#print(final_array)
#print(final_array.dtype)


"""求三点围城的三角形的外接圆圆心以及外接圆半径"""

def circumcenter(tri):
    a = tri[0]
    b = tri[1]
    c = tri[2]

    D = 2*(a[0]*(b[1]-c[1])+b[0]*(c[1]-a[1])+c[0]*(a[1]-b[1]))
    Ux = 1/D*((a[0]**2+a[1]**2)*(b[1]-c[1])+(b[0]**2+b[1]**2)*(c[1]-a[1])+(c[0]**2+c[1]**2)*(a[1]-b[1]))
    Uy = 1/D*((a[0]**2+a[1]**2)*(c[0]-b[0])+(b[0]**2+b[1]**2)*(a[0]-c[0])+(c[0]**2+c[1]**2)*(b[0]-a[0]))
    U = np.array([Ux, Uy])

    R = np.linalg.norm(U-a)
    return U, R



"""主题程序，生成三角网络"""

#计时开始
T1 = time.perf_counter()


delaunay_tri_table = [] #记录完整的符合要求的delaunay三角形索引坐标
temp_tri_table = []  #临时三角形列表
temp_edge = [] #需要删除的边表
idx_table = []  #需要删除的三角形列表对应的索引位置
centers = []  #每个三角形的外接圆圆心坐标
radius = []  #每个三角形外接圆对应的半径
idx_point = 3
for dx, dy in points_sorted:
    if dx == x_min:
        temp_tri_table.append(set([0,1,idx_point]))  #插入的三角形使用集合进行表示，因为集合可以自动去重，并且可以进行排序
        temp_tri_table.append(set([0,2,idx_point]))  #同时这里也符合整体的数据结构：列表+集合形式
        temp_tri_table.append(set([2,1,idx_point]))
        idx_point += 1
        continue


    """计算每个三角形的圆心坐标以及半径,并且在每个大for循环结束时，都要将其重新清空"""
    for temp_tri_idx in temp_tri_table:
        vertices_per_tri = final_array[list(temp_tri_idx)]
        U, R = circumcenter(vertices_per_tri)
        centers.append(U)
        radius.append(R)
        #vertices_per_tri.clear()

    """引入新的顶点进行各列表迭代"""
    for i, r in enumerate(radius):
        distance = np.sqrt((centers[i][0] - dx)**2 + (centers[i][1] - dy)**2)
        #print(distance)
        if distance > r:
            if centers[i][0] + r < dx:
                delaunay_tri_table.append(temp_tri_table[i]) #添加delaunay三角
                idx_table.append(i) #后面将delaunay三角形从临时三角形表里删去
        else:
            idx_table.append(i)
            new_set = set()
            # 生成所有包含两个元素的组合，并将每个组合转换成元组并加入新集合中
            for combo in combinations(temp_tri_table[i], 2):
                new_set.add(tuple(combo))
            #print(new_set)
            new_list = [set(t) for t in new_set]
            #print(new_list)
            temp_edge.extend(new_list)

    """对边集完全去重，只要元素重复立刻删除！！"""

    #print(temp_edge)

    unique_data = []
    for d in temp_edge:
        if d not in unique_data:
            unique_data.append(d)
        else:
            unique_data.remove(d)
    temp_edge = unique_data


    #这里我一开始写错了，我以为是去重而不是完全去重
    #temp_edge_new0= set(tuple(sorted(s)) for s in temp_edge)
    #temp_edge_new1 = [set(t) for t in temp_edge_new0]
    #temp_edge = temp_edge_new1



    #temp_edge = remove_duplicates(temp_edge)

    """删除不符合要求的临时三角形"""
    temp_tri_table_remove = [temp_tri_table[t] for t in idx_table]
    set1 = set(frozenset(i) for i in temp_tri_table)
    set2 = set(frozenset(i) for i in temp_tri_table_remove)
    result = [set(i) for i in set1.difference(set2)]  #对两个集合求差集
    temp_tri_table = result

    """将临时边集与新点重新构成三角形，并更新到临时三角形列表里"""
    new_temp_tri = [{idx_point, *d} for d in temp_edge]
    temp_tri_table.extend(new_temp_tri)

    """清空临时边表与临时索引表"""
    temp_edge = []
    idx_table = []
    if dx == x_max:
        break
    centers = []
    radius = []
    idx_point += 1




#合并临时三角形列表与delaunay表
for s in delaunay_tri_table:
    temp_tri_table.append(s)
temp_tri_table = np.array(temp_tri_table)

#将里面所有基本元素转换为列表
temp_tri_list = []
for s in temp_tri_table:
    temp_tri_list.append(list(s))

#将列表转换为ndarray矩阵方便进行判断
temp_tri_array = np.array(temp_tri_list)
#print(temp_tri_array)
temp_tri_array[temp_tri_array < 3] = 0

#构造布尔数组，记录每行是否都大于零
positive_rows = np.all(temp_tri_array > 0, axis=1)

#使用布尔索引提取符合条件的行
result = temp_tri_array[positive_rows]

"""绘图"""

#以下是使用自己的结果进行绘制比较
plt.triplot(final_array[:,0], final_array[:,1], result)
plt.plot(points_sorted[:,0], points_sorted[:,1], 'o')
T2 =time.perf_counter()
print('自己的程序运行时间:%s毫秒' % ((T2 - T1)*1000))

plt.show()


#使用sicpy中集成好的delaunay函数进行绘图并相互比较

T3 =time.perf_counter()
tri = Delaunay(points_sorted)
plt.triplot(points_sorted[:,0], points_sorted[:,1], tri.simplices)
plt.plot(points_sorted[:,0], points_sorted[:,1], '*')
T4 = time.perf_counter()
print('集成程序运行时间:%s毫秒' % ((T4 - T3)*1000))
plt.show()




