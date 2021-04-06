import telebot
import time
import os
token = "1780388562:AAEzyzS9YRCPQF6rME6A9U4lWArR6QDDYYM"
bot = telebot.TeleBot(token)

video_name = "data_files/test2.mp4"


def send_new_posts(video_name=video_name):
    channel = '-1001388181852'
    video = open(video_name, 'rb')
    bot.send_video(channel, video, timeout=10000)
    bot.send_message(channel, "{}".format(video_name))
    # bot.send_message(channel, "короче я домой)) оставлю запущенным пока")
    # Спим секунду, чтобы избежать разного рода ошибок и ограничений (на всякий случай!)
    time.sleep(1)
    return


if __name__ == '__main__':
    sent_videos = []
    while True:
        all_videos = os.listdir("output")
        video_name = all_videos.pop(0)
        video_path = os.path.join("output", video_name)
        if video_name[-3:] == "mp4" and video_name not in sent_videos:
            send_new_posts(video_path)
            sent_videos.append(video_name)
        bot.polling(none_stop=True, timeout=333)
