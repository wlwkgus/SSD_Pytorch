from utils import label


class StoreHierarchicalManager(object):
    LABEL = label.LABEL['area']['store']

    def __init__(self):
        self.all_hash_key = self.get_all_hash_key()
        one_depth = set()
        for d in self.all_hash_key:
            one_depth.add(d[0])
        one_depth = list(one_depth)

        two_depth = dict()
        for d in one_depth:
            two_depth[d] = set()
        for d in self.all_hash_key:
            two_depth[d[0]].add(d[1])

        for d in one_depth:
            two_depth[d] = list(two_depth[d])

        three_depth = dict()
        for d1, d2_list in two_depth.items():
            three_depth[d1] = dict()
            for d2 in d2_list:
                three_depth[d1][d2] = set()

        for d in self.all_hash_key:
            three_depth[d[0]][d[1]].add(d[2])

        for d1, d2_list in two_depth.items():
            for d2 in d2_list:
                three_depth[d1][d2] = list(three_depth[d1][d2])

        self.one_depth = one_depth
        self.two_depth = two_depth
        self.three_depth = three_depth

    def get_hash_key(self, key):
        return self._get_key(key, self.LABEL)

    def _get_key(self, k, d, depth=list()):
        if type(d) == list:
            for value in d:
                if k == value:
                    return depth + [k]
            return None
        else:
            for key in d.keys():
                ret = self._get_key(k, d[key], depth + [key])
                if ret is not None:
                    return ret

    def get_all_hash_key(self):
        return self._get_h_key(self.LABEL)

    def _get_h_key(self, d, depth=list()):
        if type(d) == list:
            ret = list()
            for value in d:
                ret.append(depth + [value])
            return ret
        else:
            ret = list()
            for key in d.keys():
                ret += self._get_h_key(d[key], depth + [key])
            return ret

    def get_output_length(self, obj):
        s = 0
        if type(obj) == list:
            return len(obj)
        else:
            for key, item in obj.items():
                s += self.get_output_length(item)
            return s

    def get_output_index(self, key, obj, cumsum=0):
        if type(obj) == list:
            for i, val in enumerate(obj):
                if val == key:
                    return cumsum, cumsum + i, cumsum + len(obj)
            return None
        else:
            partial_cumsum = 0
            for k, item in obj.items():
                ret = self.get_output_index(key, item, cumsum + partial_cumsum)
                if ret is not None:
                    return ret
                partial_cumsum += self.get_output_length(item)

    def get_three_depth_index_by_key(self, key):
        three_depth_list = self.get_hash_key(key)

        return [
            self.get_output_index(three_depth_list[0], self.one_depth),
            self.get_output_index(three_depth_list[1], self.two_depth),
            self.get_output_index(three_depth_list[2], self.three_depth)
        ]


class RestHierarchicalManager(object):
    LABEL = LabelManager.LABEL['area']
    LABEL.pop('store')

    def __init__(self):
        self.all_hash_key = self.get_all_hash_key()
        one_depth = set()
        for d in self.all_hash_key:
            one_depth.add(d[0])
        one_depth = list(one_depth)

        two_depth = dict()
        for d in one_depth:
            two_depth[d] = set()
        for d in self.all_hash_key:
            two_depth[d[0]].add(d[1])

        for d in one_depth:
            two_depth[d] = list(two_depth[d])

        self.one_depth = one_depth
        self.two_depth = two_depth

    def get_hash_key(self, key):
        return self._get_key(key, self.LABEL)

    def _get_key(self, k, d, depth=list()):
        if type(d) == list:
            for value in d:
                if k == value:
                    return depth + [k]
            return None
        else:
            for key in d.keys():
                ret = self._get_key(k, d[key], depth + [key])
                if ret is not None:
                    return ret

    def get_all_hash_key(self):
        return self._get_h_key(self.LABEL)

    def _get_h_key(self, d, depth=list()):
        if type(d) == list:
            ret = list()
            for value in d:
                ret.append(depth + [value])
            return ret
        else:
            ret = list()
            for key in d.keys():
                ret += self._get_h_key(d[key], depth + [key])
            return ret

    def get_output_length(self, obj):
        s = 0
        if type(obj) == list:
            return len(obj)
        else:
            for key, item in obj.items():
                s += self.get_output_length(item)
            return s

    def get_output_index(self, key, obj, cumsum=0):
        if type(obj) == list:
            for i, val in enumerate(obj):
                if val == key:
                    return cumsum, cumsum + i, cumsum + len(obj)
            return None
        else:
            partial_cumsum = 0
            for k, item in obj.items():
                ret = self.get_output_index(key, item, cumsum + partial_cumsum)
                if ret is not None:
                    return ret
                partial_cumsum += self.get_output_length(item)

    def get_two_depth_index_by_key(self, key):
        three_depth_list = self.get_hash_key(key)

        return [
            self.get_output_index(three_depth_list[0], self.one_depth),
            self.get_output_index(three_depth_list[1], self.two_depth),
        ]


if __name__ == '__main__':
    s_manager = StoreHierarchicalManager()
    one_depth_length = s_manager.get_output_length(s_manager.one_depth)
    two_depth_length = s_manager.get_output_length(s_manager.two_depth)
    three_depth_length = s_manager.get_output_length(s_manager.three_depth)

    print(s_manager.get_three_depth_index_by_key('접이식테이블'))
    print(s_manager.get_three_depth_index_by_key('트리'))

    r_manager = RestHierarchicalManager()

    one_depth_length = s_manager.get_output_length(s_manager.one_depth)
    two_depth_length = s_manager.get_output_length(s_manager.two_depth)

    print(r_manager.get_two_depth_index_by_key("대리석타일"))
    print(r_manager.get_two_depth_index_by_key("1단 벽선반"))
