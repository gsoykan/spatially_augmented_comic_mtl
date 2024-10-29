def faster_rcnn_collate_fn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*filter(lambda x: x is not None, batch)))
