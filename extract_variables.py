import pandas as pd
from performance_variables import extract_triage_variables, extract_triage_variables_789, unfreeze_count
from spatial_variables import extract_bypass_variables, extract_revisit_variables, extract_visit_variables
from temporal_variables import extract_sprint_variables, extract_time2completion_variables
from competency_variables import extract_competency_variables
from file_utils import json2df, json2df_2, json2df_dac
from teaming_variables import extract_proximity_variables, extract_role_variables, extract_team_role_variables, find_team_info
from plot_util import visualization_role, visualization_dac


def process_trial(data_file, dac_file, team_id, building):
     
    df = json2df_2(data_file)
    if dac_file != None:
        df_dac = json2df_dac(dac_file)
    # victims = pd.DataFrame(df[(df['msg.sub_type'] == 'Mission:VictimList')]['data.mission_victim_list'].item())
    # victims = pd.read_csv(building.victims_file)
    df_export = df[(df['msg.sub_type'] == 'start')]
    df_pre = df[(df['data.mission_timer'] == "Mission Timer not initialized.")&(df['msg.sub_type'] == 'Event:RoleSelected')]

    players_info = df_export['data.client_info'].to_list()[0]
    if len(players_info) > 0:
        for i in range(len(players_info)):
            if players_info[i]['callsign'] in ['Alpha', 'Red']:
                p1 = players_info[i]['playername']
                # p1 = players_info[i]['participantid']
                p1_callsign = players_info[i]['callsign']
                p1_id = players_info[i]['participantid']
                if 'markerblocklegend' in players_info[i]:
                    p1_marker_legend = players_info[i]['markerblocklegend']
                else:
                    p1_marker_legend = 'None'
                if 'staticmapversion' in players_info[i]:
                    p1_map_version = players_info[i]['staticmapversion']
                else:
                    p1_map_version = 'None'
            elif players_info[i]['callsign'] in ['Bravo', 'Green']:
                p2 = players_info[i]['playername']
                # p2 = players_info[i]['participantid']
                p2_callsign = players_info[i]['callsign']
                p2_id = players_info[i]['participantid']
                if 'markerblocklegend' in players_info[i]:
                    p2_marker_legend = players_info[i]['markerblocklegend']
                else:
                    p2_marker_legend = 'None'
                if 'staticmapversion' in players_info[i]:
                    p2_map_version = players_info[i]['staticmapversion']
                else:
                    p2_map_version = 'None'
            else:
                p3 = players_info[i]['playername']
                # p3 = players_info[i]['participantid']
                p3_callsign = players_info[i]['callsign']
                p3_id = players_info[i]['participantid']
                if 'markerblocklegend' in players_info[i]:
                    p3_marker_legend = players_info[i]['markerblocklegend']
                else:
                    p3_marker_legend = 'None'
                if 'staticmapversion' in players_info[i]:
                    p3_map_version = players_info[i]['staticmapversion']
                else:
                    p3_map_version = 'None'
    else:
        p1 = -99
        p2 = -99
        p3 = -99
        p1_callsign = -99
        p2_callsign = -99
        p3_callsign = -99
        p1_marker_legend = -99
        p2_marker_legend = -99
        p3_marker_legend = -99
        p1_map_version = -99
        p2_map_version = -99
        p3_map_version = -99
        players_id = df_export['data.subjects'].to_list()[0]
        p1_id = players_id[0]
        p2_id = players_id[1]
        p3_id = players_id[2]

    if team_id == 'TM000013':
        p1 = 'joseph6200'

    if team_id =='TM000014':
        p2 = 'ChunkyVortex'
        p3 = 'KevinsWither'

    if team_id == 'TM000011':
        p1 = 'xVisualize'
        p2 = 'yingura'
        p3 = 'MelloD'

    mission_times = df[df['msg.sub_type'] == 'Event:MissionState']['msg.timestamp'].tolist()
    if len(mission_times) > 0:
        mission_start = mission_times[0]
        df = df[df['msg.timestamp'] > mission_start]

    df['msg.timestamp'] = df['msg.timestamp'].dt.tz_localize(None)
    df = df[df['data.mission_timer'] != "Mission Timer not initialized."]
    df = df[df['data.mission_timer'].notna()]
    df = df.sort_values(by='msg.timestamp')

    # df = df[df['data.participant_id'].isin([p1, p2, p3])]
    # df_1 = df[(df['data.participant_id'] == p1)]  # player_1
    # df_2 = df[(df['data.participant_id'] == p2)]  # player_2
    # df_3 = df[(df['data.participant_id'] == p3)]  # player_3
    # df_1_pre = df_pre[(df_pre['data.participant_id'] == p1)]
    # df_2_pre = df_pre[(df_pre['data.participant_id'] == p2)]
    # df_3_pre = df_pre[(df_pre['data.participant_id'] == p3)]
    
    df = df[df['data.playername'].isin([p1, p2, p3])]
    df_1 = df[(df['data.playername'] == p1)]  # player_1
    df_2 = df[(df['data.playername'] == p2)]  # player_2
    df_3 = df[(df['data.playername'] == p3)]  # player_3
    df_1_pre = df_pre[(df_pre['data.playername'] == p1)]
    df_2_pre = df_pre[(df_pre['data.playername'] == p2)]
    df_3_pre = df_pre[(df_pre['data.playername'] == p3)]

    building_zone = pd.read_csv(building.zones_file)
    # zone_victims = pd.read_csv(building.zones_victim_file)
    # victims = pd.read_csv(building.victims_file)
    victims = pd.read_csv(building.victim_class_file)
    rubble_classes = pd.read_csv(building.rubble_class_file)

    unfreeze_1, unfreeze_2, unfreeze_3 = unfreeze_count(df,p1,p2,p3)

    visit_variables_team = extract_visit_variables(df, building_zone, victims, p1, p2, p3, 'Team_')
    visit_variables_2 = extract_visit_variables(df_2, building_zone, victims, p1, p2, p3, 'Player2_')
    visit_variables_1 = extract_visit_variables(df_1, building_zone, victims, p1, p2, p3,'Player1_')
    visit_variables_3 = extract_visit_variables(df_3, building_zone, victims,p1, p2, p3, 'Player3_')

    triage_variables_2 = extract_triage_variables(df_2, victims, rubble_classes, 'Player2_')
    triage_variables_1 = extract_triage_variables(df_1, victims, rubble_classes, 'Player1_')

    triage_variables_3 = extract_triage_variables(df_3, victims, rubble_classes, 'Player3_')
    triage_variables_team = extract_triage_variables(df, victims, rubble_classes, 'Team_')

    triage_variables_team['Distance_total'] = (triage_variables_1['Distance_total']+triage_variables_2['Distance_total']+triage_variables_3['Distance_total'])[0].round(2)

    role_variables_1, time_stamp_list_1, role_list_1 = extract_role_variables(df_1, df_1_pre, 'Player1_')
    role_variables_2, time_stamp_list_2, role_list_2 = extract_role_variables(df_2, df_2_pre, 'Player2_')
    role_variables_3, time_stamp_list_3, role_list_3 = extract_role_variables(df_3, df_3_pre, 'Player3_')
    role_variables_team, role_redundant_1, role_redundant_2, role_redundant_3 = extract_team_role_variables(df, p1, p2,
                                                                                                            p3, 'Team_')
    # visualization_role(df, df_1, df_2, df_3, time_stamp_list_1, role_list_1, time_stamp_list_2, role_list_2, time_stamp_list_3, role_list_3, team_id)


    proximity_variables_1 = extract_proximity_variables(df, team_id, p1, p2, p3, time_stamp_list_1, role_list_1, time_stamp_list_2, role_list_2,time_stamp_list_3, role_list_3, 'Player1_')
    proximity_variables_2 = extract_proximity_variables(df, team_id, p1, p2, p3, time_stamp_list_1, role_list_1, time_stamp_list_2, role_list_2,time_stamp_list_3, role_list_3, 'Player2_')
    proximity_variables_3 = extract_proximity_variables(df, team_id, p1, p2, p3, time_stamp_list_1, role_list_1, time_stamp_list_2, role_list_2,time_stamp_list_3, role_list_3, 'Player3_')
    proximity_variables_team = extract_proximity_variables(df, team_id, p1, p2, p3, time_stamp_list_1, role_list_1, time_stamp_list_2, role_list_2,time_stamp_list_3, role_list_3, 'Team_')



    # visualization_dac(df, df_1, df_2, df_3, time_stamp_list_1, role_list_1, time_stamp_list_2, role_list_2, time_stamp_list_3, role_list_3, team_id,
    #                   df_dac, p1, p2, p3)

    p1_data = [[p1_id, p1, p1_callsign, p1_marker_legend, p1_map_version]]
    p1_df = pd.DataFrame.from_records(p1_data, columns=['player_id', 'player_name', 'Callsign', 'Marker_block_legend', 'Map_version'])
    p2_data = [[p2_id, p2, p2_callsign, p2_marker_legend, p2_map_version]]
    p2_df = pd.DataFrame.from_records(p2_data, columns=['player_id', 'player_name', 'Callsign', 'Marker_block_legend', 'Map_version'])
    p3_data = [[p3_id, p3, p3_callsign, p3_marker_legend, p3_map_version]]
    p3_df = pd.DataFrame.from_records(p3_data, columns=['player_id', 'player_name', 'Callsign', 'Marker_block_legend', 'Map_version'])

    # p1_data = [[p1_id, p1, p1_callsign]]
    # p1_df = pd.DataFrame.from_records(p1_data, columns=['player_id', 'player_name', 'Callsign'])
    # p2_data = [[p2_id, p2, p2_callsign]]
    # p2_df = pd.DataFrame.from_records(p2_data, columns=['player_id', 'player_name', 'Callsign'])
    # p3_data = [[p3_id, p3, p3_callsign]]
    # p3_df = pd.DataFrame.from_records(p3_data, columns=['player_id', 'player_name', 'Callsign'])

    # team_distance_df = pd.DataFrame.from_records([[team_distance_total]],
    #                                          columns=['Distance_total'])

    variables_p1 = pd.concat([p1_df, triage_variables_1, visit_variables_1, proximity_variables_1, role_variables_1, role_redundant_1, unfreeze_1], axis=1)
    variables_p2 = pd.concat([p2_df, triage_variables_2, visit_variables_2, proximity_variables_2, role_variables_2, role_redundant_2, unfreeze_2], axis=1)
    variables_p3 = pd.concat([p3_df, triage_variables_3, visit_variables_3, proximity_variables_3, role_variables_3, role_redundant_3, unfreeze_3], axis=1)
    variables_team = pd.concat([triage_variables_team, visit_variables_team, proximity_variables_team, role_variables_team], axis=1)

    return variables_p1, variables_p2, variables_p3, variables_team
    # return 0


def process_competency(data_file, tag):
    df = json2df(data_file)
    df['msg.timestamp'] = pd.to_datetime(df['msg.timestamp'])
    df['msg.timestamp'] = df['msg.timestamp'].dt.tz_localize(None)
    variables = extract_competency_variables(df, tag)
    return variables





