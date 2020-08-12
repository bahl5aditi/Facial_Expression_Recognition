from flask import Flask,render_template,Response
from Camera import videoCamera

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(Camera):
    while True:
        frame=Camera.get_frame()
        yield(b'--frame\r\r'
              b'Content-Type:image/jpeg\r\n\r\n'+frame+b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(videoCamera),
                    mimetype='multipart/x-mixed-replace;boundary=frame')

if __name__== "main":
    app.run(host='localhost',port=5000,debug=True)

