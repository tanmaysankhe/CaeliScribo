
from flask import Flask, render_template, request, Response, jsonify
from camera import VideoCamera
import cv2
import time
from markov import markov_here


app = Flask(__name__)

written_content = ""
current_word = ""
prev_word = '-'
#prev_5_time = time.time()
prev_state = 0
pred_on_off = 0
new_markov = 0
top3 = ['Hello', 'Hi', 'I'] 


@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    c = 1
    global written_content, prev_word, prev_state, new_markov, current_word,pred_on_off, top3 #prev_5_time
    while True:
        frame = camera.get_frame()
        #print(c)

        if (pred_on_off == 0):
            if prev_state == 5 and frame[2] != 5:
                print(frame[2])
                prev_state = frame[2]

            new_word = (frame[1])
            if frame[2] == 5 and prev_state != 5:
                print(frame[2])
                prev_state = 5
                #print("5555555" + str (time.time() - prev_5_time))
                #if time.time() - prev_5_time > 3:
                #prev_5_time = time.time()
                if len(current_word) != 0:
                    current_word = current_word[:-1]
                else :
                    written_content = written_content[:-1]
                new_markov = 1
                print('Erased!') 

            if frame[2] == 3 and prev_state != 3:
                print(frame[2])
                prev_state = 3
                written_content += " " + current_word
                current_word = ""

            if frame[2] == 4 and prev_state != 4:
                print(frame[2])
                pred_on_off = 1



            elif (prev_word != new_word and new_word != ""):
                prev_word = new_word
                current_word += new_word
                new_markov = 1

        else:
            if (frame[2] == 4 ):
                pass
            else:
                
                written_content += current_word
                current_word = ""
                pred_on_off = 0
                

                if frame[2] == 1:
                    written_content += " " + top3[0]
                    top3 = markov_here(top3[0])
                

                elif frame[2] == 2:
                    written_content += " " + top3[1]
                    top3 = markov_here(top3[1])

                
                elif frame[2] == 3:
                    written_content += " " + top3[2]
                    top3 = markov_here(top3[2])
            
                

        c +=1
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame[0] + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/word_fetch', methods=['GET', 'POST'])
def word_fetch():
    json = request.get_json()
    global written_content, current_word

    return jsonify(written_content = written_content + current_word)

@app.route('/render_word')
def render_word():
    return render_template('word.html')

@app.route('/three')
def open_Three():
    return render_template('three.html')

@app.route('/markov_man', methods=['POST'])
def markov_buddy():
    global written_content, new_markov, top3, current_word
    if new_markov == 1:
        top3 = markov_here(current_word)
        print(top3)
        new_markov = 0
    #print(top3['i'][0])
    if ( top3 == None or len(top3) < 3):
        top3 = ['Indeed', 'Well', 'I']

    return jsonify (one = top3[0], two = top3[1], three = top3[2])

if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=5000,debug=True)