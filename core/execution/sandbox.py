import signal, contextlib
class TimeoutError(Exception): pass
@contextlib.contextmanager
def time_limit(seconds:int):
    def h(signum, frame): raise TimeoutError('execution timed out')
    old=signal.signal(signal.SIGALRM, h); signal.alarm(seconds)
    try: yield
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, old)
