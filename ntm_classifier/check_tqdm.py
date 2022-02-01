def tqdm_check():
    try:
        get_ipython().__class__.__name__  # noqa
        return True

    except BaseException:
        if __name__ == '__main__':
            return True
    return False
