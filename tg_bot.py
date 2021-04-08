import telebot
import time
import os
token = "1780388562:AAEzyzS9YRCPQF6rME6A9U4lWArR6QDDYYM"
bot = telebot.TeleBot(token)


def send_new_posts(videoname, actionname):
    channel = '-1001388181852'
    video_path = os.path.join("output", videoname)
    video = open(video_path, 'rb')
    video_time = videoname[:-4].split()
    bot.send_message(channel, "Человек {} в {}:{}:{}".format(actionname, video_time[0], video_time[1], video_time[2]))
    bot.send_video(channel, video, timeout=50)
    # bot.send_message(channel, "короче я домой)) оставлю запущенным пока")
    # Спим секунду, чтобы избежать разного рода ошибок и ограничений (на всякий случай!)
    # time.sleep(1)
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
        # bot.polling(none_stop=True, timeout=333)
