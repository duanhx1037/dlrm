def write2rankfile(_str, rank_id, _mode='w+'):
    '''
    This function writes certain string to corressponding GPU's print file
    '''
    with open(f'dist_GPUs_print/rank_{rank_id}.txt', mode=_mode) as f:
        f.write(_str)
    