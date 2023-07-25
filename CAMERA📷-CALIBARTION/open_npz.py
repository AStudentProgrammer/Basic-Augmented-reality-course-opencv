from numpy import load

data = load('calib_data/MultiMatrix.npz')
lst = data.files
# for item in lst:
#     print(item)
#     print(data[item])
print(data["camMatrix"])