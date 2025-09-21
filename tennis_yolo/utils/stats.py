import pandas as pd
import numpy as np
from copy import deepcopy
from pathlib import Path

from utils import DATA_DIR

def compute_player_stats(player_dt, ball_dt, ball_hit, court_kp, fps, save = False):
    player_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,

        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
    }]

    court_kp = court_kp.reshape(-1, 2) #to use (x, y)
    COURT_WIDTH_METERS = 10.97 
    COURT_HEIGHT_METERS = 23.78

    for i in range(len(ball_hit) - 1):
        start_f = ball_hit[i] #start (x, y)
        end_f = ball_hit[i + 1] #end (x, y)
        dt = (end_f - start_f) / fps

        #distance and speed
        ball_start = get_center_of_bbox(ball_dt[start_f][1])
        ball_end = get_center_of_bbox(ball_dt[end_f][1])

        ball_dist_pix = measure_distance(ball_start, ball_end)
        ball_dist_m = convert_pixel_distance_to_meters(
            ball_dist_pix,
            COURT_WIDTH_METERS,
            abs(court_kp[0][0] - court_kp[1][0])  
        )
        ball_speed = (ball_dist_m / dt) * 3.6  # km/h

        #player 1
        players_start = player_dt[start_f]
        hitter_id = min(players_start.keys(),
                        key=lambda pid: measure_distance(
                            get_center_of_bbox(players_start[pid]), ball_start))

        #player 2
        opponent_id = 1 if hitter_id == 2 else 2
        opp_start = get_center_of_bbox(player_dt[start_f][opponent_id])
        opp_end = get_center_of_bbox(player_dt[end_f][opponent_id])

        opp_dist_pix = measure_distance(opp_start, opp_end)
        opp_dist_m = convert_pixel_distance_to_meters(
            opp_dist_pix,
            COURT_WIDTH_METERS,
            abs(court_kp[0][0] - court_kp[1][0])
        )
        opp_speed = (opp_dist_m / dt) * 3.6

        current_stats = deepcopy(player_stats_data[-1])
        current_stats['frame_num'] = start_f

        current_stats[f'player_{hitter_id}_number_of_shots'] += 1
        current_stats[f'player_{hitter_id}_total_shot_speed'] += ball_speed
        current_stats[f'player_{hitter_id}_last_shot_speed'] = ball_speed

        current_stats[f'player_{opponent_id}_total_player_speed'] += opp_speed
        current_stats[f'player_{opponent_id}_last_player_speed'] = opp_speed

        player_stats_data.append(current_stats)

    df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(ball_dt)))})
    df = pd.merge(frames_df, df, on='frame_num', how='left')
    df = df.ffill()

    df['player_1_average_shot_speed'] = df['player_1_total_shot_speed'] / df['player_1_number_of_shots'].replace(0, 1)
    df['player_2_average_shot_speed'] = df['player_2_total_shot_speed'] / df['player_2_number_of_shots'].replace(0, 1)
    df['player_1_average_player_speed'] = df['player_1_total_player_speed'] / df['player_1_number_of_shots'].replace(0, 1)
    df['player_2_average_player_speed'] = df['player_2_total_player_speed'] / df['player_2_number_of_shots'].replace(0, 1)

    court_kp = court_kp.reshape(-1)

    if save:
        results_path = Path(DATA_DIR) / "results" / "player_stats.csv"
        df.to_csv(results_path, index=False)

    return df

def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def convert_pixel_distance_to_meters(pixel_distance, refrence_height_in_meters, refrence_height_in_pixels):
    return (pixel_distance * refrence_height_in_meters) / refrence_height_in_pixels