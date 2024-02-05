import time
import functools


def timeit(func):
    """Decorator to measure the execution time of a function in milliseconds."""

    @functools.wraps(func)  # Preserve the metadata of the original function
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.time()  # Record end time
        elapsed_time = (end_time - start_time)  # Calculate elapsed time in milliseconds
        print(f"Function '{func.__name__}' executed in {elapsed_time:.2f} s")
        return result

    return wrapper
