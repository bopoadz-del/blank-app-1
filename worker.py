import os
from rq import Worker, Queue, Connection
import redis
listen=['default']
conn=redis.from_url(os.getenv('REDIS_URL','redis://redis:6379/0'))
if __name__=='__main__':
    with Connection(conn): Worker(list(map(Queue, listen))).work()
