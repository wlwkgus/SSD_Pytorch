from utils.bucket_hierarchical import StoreHierarchicalManager, RestHierarchicalManager


class DepthManager(object):
    def __init__(self):
        self.s_h_manager = StoreHierarchicalManager()
        self.r_h_manager = RestHierarchicalManager()

    def get_depth_by_keyword(self, keyword):
        depth = self.s_h_manager.get_three_depth_index_by_key(keyword)
        if depth is None:
            depth = self.r_h_manager.get_two_depth_index_by_key(keyword)
        if depth is None:
            raise Exception("Should not be here.")
        return depth

    def get_chunk_index_and_label(self, keyword):
        if keyword == 'background':
            return [(0, 0)]
        hash_key = self.r_h_manager.get_hash_key(keyword)
        if hash_key is None:
            hash_key = ['store'] + self.s_h_manager.get_hash_key(keyword)
        ret = list()
        for key in hash_key:
            numsum = 0
            key_index = self.all_list_of_labels.index(key)
            for i, num in enumerate(self.all_list_of_num_classes):
                if numsum <= key_index < numsum + num:
                    ret.append(
                        (i, key_index - numsum)
                    )
                    break
                numsum += num
        return ret


    @property
    def all_list_of_num_classes(self):
        # Background label
        ret = list(
            self.r_h_manager.yield_all_list_of_num_classes()
        ) + list(
            self.s_h_manager.yield_all_list_of_num_classes()
        )
        ret[0] += 1
        return ret

    @property
    def all_list_of_labels(self):
        # Background
        return ['background'] + list(
            self.r_h_manager.yield_all_labels()
        ) + list(
            self.s_h_manager.yield_all_labels()
        )

if __name__ == '__main__':
    d_manager = DepthManager()
    print(d_manager.all_list_of_num_classes)
    import numpy as np
    print(np.asarray(d_manager.all_list_of_num_classes).sum())
    print(d_manager.all_list_of_labels)
    print(d_manager.get_chunk_index_and_label("3단 벽선반"))
    print(d_manager.get_chunk_index_and_label("background"))
    print(d_manager.get_chunk_index_and_label("빈백소파"))
