
def get_mb_tensor(tensor):
    nbytes = tensor.element_size() * tensor.nelement()
    mb = nbytes /1000000
    return mb