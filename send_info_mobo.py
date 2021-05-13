import os
import socket

import telebot

HOST = "localhost"
PORT = 8083
token = "1868509329:AAHGNVxAuV2oCl_cf9O87jaYP4t7b0jRY7w"

bot = telebot.TeleBot(token)
sent_videos = set()


def send_new_posts(videoname, actionname):
    # channel = '-1001388181852'
    channel = '-1001399933919'

    video_path = os.path.join("output", videoname)
    video = open(video_path, 'rb')
    video_time = videoname[:-4].split()
    text_caption = "Человек {} в {}:{}:{}".format(actionname, video_time[0], video_time[1], video_time[2])
    # bot.send_message(chat_id=channel,text="Человек {} в {}:{}:{}".
    #                  format(actionname, video_time[0], video_time[1], video_time[2]))
    # bot.send_document(chat_id=channel, data=video, caption=text_caption, timeout=50)
    bot.send_video(chat_id=channel, data=video, caption=text_caption, timeout=50)

    return

def send_message(current_date, counter_in, counter_out):
    channel = '-1001399933919'
    msg_tosend = "сегодня {}: зашло {}, вышло {}".format(current_date, counter_in, counter_out)
    bot.send_message(chat_id=channel,text=msg_tosend)


# with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#     s.bind((HOST, PORT))
#     s.listen()
#     conn, addr = s.accept()
#     with conn:
#         print('Connected by', addr)
#         while True:
#             data = conn.recv(100)
#             video_data = data.decode("utf-8").split(":")
#             conn.sendall(bytes('received: ' + video_data[0] + video_data[1], "utf-8"))
#             print(video_data)
#             send_new_posts(video_data[0], video_data[1])
# send_message(0, 0)

