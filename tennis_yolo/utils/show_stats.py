import numpy as np
import cv2 as cv

def draw_player_stats(output_video_frames, player_stats, width_st = 350, height_st = 230):

    for index, row in player_stats.iterrows():
        player_1_shot_speed = row['player_1_last_shot_speed']
        player_2_shot_speed = row['player_2_last_shot_speed']
        player_1_speed = row['player_1_last_player_speed']
        player_2_speed = row['player_2_last_player_speed']

        avg_player_1_shot_speed = row['player_1_average_shot_speed']
        avg_player_2_shot_speed = row['player_2_average_shot_speed']
        avg_player_1_speed = row['player_1_average_player_speed']
        avg_player_2_speed = row['player_2_average_player_speed']

        frame = output_video_frames[index]

        width = width_st
        height = height_st

        space = 20
        start_x = frame.shape[1] - width - space 
        start_y = space                            
        end_x = start_x + width
        end_y = start_y + height

        overlay = frame.copy()
        cv.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        alpha = 0.5
        cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        output_video_frames[index] = frame

        font = cv.FONT_HERSHEY_SIMPLEX

        text = "     Player 1     Player 2"
        cv.putText(output_video_frames[index], text, (start_x + 60, start_y + 30), font, 0.6, (255, 255, 255), 2)

        text = "Shot Speed"
        cv.putText(output_video_frames[index], text, (start_x + 10, start_y + 70), font, 0.45, (255, 255, 255), 1)
        text = f"{player_1_shot_speed:.1f} km/h    {player_2_shot_speed:.1f} km/h"
        cv.putText(output_video_frames[index], text, (start_x + 120, start_y + 70), font, 0.5, (255, 255, 255), 2)

        text = "Player Speed"
        cv.putText(output_video_frames[index], text, (start_x + 10, start_y + 110), font, 0.45, (255, 255, 255), 1)
        text = f"{player_1_speed:.1f} km/h    {player_2_speed:.1f} km/h"
        cv.putText(output_video_frames[index], text, (start_x + 120, start_y + 110), font, 0.5, (255, 255, 255), 2)

        text = "avg. S. Speed"
        cv.putText(output_video_frames[index], text, (start_x + 10, start_y + 150), font, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_shot_speed:.1f} km/h    {avg_player_2_shot_speed:.1f} km/h"
        cv.putText(output_video_frames[index], text, (start_x + 120, start_y + 150), font, 0.5, (255, 255, 255), 2)

        text = "avg. P. Speed"
        cv.putText(output_video_frames[index], text, (start_x + 10, start_y + 190), font, 0.45, (255, 255, 255), 1)
        text = f"{avg_player_1_speed:.1f} km/h    {avg_player_2_speed:.1f} km/h"
        cv.putText(output_video_frames[index], text, (start_x + 120, start_y + 190),font, 0.5, (255, 255, 255), 2)

    return output_video_frames