import uuid
from starlette.middleware.base import BaseHTTPMiddleware
class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        request.state.request_id=str(uuid.uuid4())
        resp=await call_next(request)
        resp.headers['X-Request-ID']=request.state.request_id
        return resp
