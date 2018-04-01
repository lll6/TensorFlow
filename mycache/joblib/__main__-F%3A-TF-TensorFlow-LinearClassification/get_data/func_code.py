# first line: 11
@mem.cache
def get_data():
    data = load_svmlight_file('F:/MLworkplace/australian_scale')
    return data[0], data[1]
