# from celery import Celery
#
# app = Celery('hello_world', broker='pyamqp://guest@localhost:1234//')
#
# @app.task
# def add(x, y):
#     return x + y

import socket
import time
import telebot
import os
HOST = "localhost"
PORT = 8084

token = "1780388562:AAEzyzS9YRCPQF6rME6A9U4lWArR6QDDYYM"
bot = telebot.TeleBot(token)
sent_videos = set()


def send_new_posts(videoname, actionname):
    channel = '-1001388181852'
    video_path = os.path.join("output", videoname)
    video = open(video_path, 'rb')
    video_time = videoname[:-4].split()
    bot.send_message(channel,
                     "Человек {} в {}:{}:{}".format(actionname, video_time[0], video_time[1], video_time[2]))
    bot.send_video(channel, video, timeout=50)
    # bot.send_message(channel, "короче я домой)) оставлю запущенным пока")
    # Спим секунду, чтобы избежать разного рода ошибок и ограничений (на всякий случай!)
    # time.sleep(1)
    return

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print('Connected by', addr)
        while True:
            data = conn.recv(100)
            video_data = data.decode("utf-8").split(":")
            print(video_data)
            send_new_posts(video_data[0], video_data[1])
            conn.sendall(bytes(str(data),"utf-8"))
            # time.sleep(10)

# add.delay(1,2)