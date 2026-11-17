"""Latest-frame MJPEG preview, with JPEG encoding outside control/ROS callbacks."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

class Preview:
    def __init__(self,host='127.0.0.1',port=8765):
        self.condition=threading.Condition();self.frame=None;self.jpeg=None;self.sequence=0
        self.closed=False;self.stop_event=threading.Event()
        owner=self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_GET(self):
                if self.path=='/':
                    body=b'<!doctype html><meta charset="utf-8"><title>Nova5 Tongji preview</title><body style="background:#111;color:white;font-family:sans-serif"><h2>Nova5 / Tongji EKF</h2><p>Green: detected | Cyan: current | Yellow: 50 ms | Magenta: 100 ms</p><img style="max-width:100%" src="/stream.mjpg">'
                    self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
                    self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
                    return
                if self.path!='/stream.mjpg':self.send_error(404);return
                self.send_response(200);self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
                self.send_header('Cache-Control','no-store');self.end_headers();seq=-1
                try:
                    while not owner.closed:
                        with owner.condition:
                            owner.condition.wait_for(lambda:owner.closed or (owner.jpeg is not None and owner.sequence!=seq),timeout=5)
                            if owner.closed:return
                            if owner.jpeg is None or owner.sequence==seq:continue
                            data,seq=owner.jpeg,owner.sequence
                        self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(data)).encode()+b'\r\n\r\n'+data+b'\r\n')
                        self.wfile.flush()
                except (OSError,TimeoutError):pass
        self.server=ThreadingHTTPServer((host,port),Handler);self.server.daemon_threads=True
        self.worker=threading.Thread(target=self.encode,daemon=True)
        self.http=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.worker.start();self.http.start()

    def publish(self,image):
        with self.condition:self.frame=image.copy();self.condition.notify_all()

    def encode(self):
        import cv2
        while not self.closed:
            with self.condition:
                self.condition.wait_for(lambda:self.frame is not None or self.closed,timeout=.5)
                if self.closed:return
                frame=self.frame;self.frame=None
            if frame is None:continue
            h,w=frame.shape[:2]
            if w>960:frame=cv2.resize(frame,(960,round(h*960/w)))
            ok,jpeg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,75])
            if ok:
                with self.condition:
                    self.jpeg=jpeg.tobytes();self.sequence+=1;self.condition.notify_all()
            # Preview has its own 10 Hz budget; no sleeping in vision/control callbacks.
            self.stop_event.wait(.1)

    def close(self):
        self.stop_event.set()
        with self.condition:self.closed=True;self.condition.notify_all()
        self.server.shutdown();self.server.server_close()
        self.worker.join(timeout=2);self.http.join(timeout=2)
