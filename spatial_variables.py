import pandas as pd
import datetime
from datetime import timedelta
import numpy as np
from collections import defaultdict

def assign_zone(player_x, player_z, zone_coords, padding=0):
    """
    Based on coordination of the player, finds which zone is the player in
    :param player_x: data.x
    :param player_z: data.z
    :param zone_coords: coordinations of all zones in the map
    :return: the zone number if the x and z are within the map zones and -1 if not
    """
    zones = []
    for zone in zone_coords:
        [zone_number, xtl, xbr, ztl, zbr] = zone
        if padding > 0:
            xtl = xtl - padding
            xbr = xbr + padding
            ztl = ztl - padding
            zbr = zbr + padding
        if (xtl <= player_x <= xbr) and (ztl <= player_z <= zbr):
            zones.append(zone_number)
        else:
            continue
    if len(zones) == 0:
        return -1
    else:
        return zones[0]
        # key = max(list(zones.keys()))


def assign_victim_zone(victim_x, victim_z, victims):
    victim_zone = victims[(victims['Xcoords'] == victim_x) & (victims['ZCoords'] == victim_z)]['Zone'].values[0]
    return victim_zone


def get_victims_per_zone(victims):
    victims_per_zone = victims.groupby('Zone').size().to_dict()
    return victims_per_zone


def get_green_victims_per_zone(victims):
    victims_per_zone = victims[victims['VictimColor'] == 'Green'].groupby('Zone').size().to_dict()
    return victims_per_zone


def get_yellow_victims_per_zone(victims):
    victims_per_zone = victims[victims['VictimColor'] == 'Yellow'].groupby('Zone').size().to_dict()
    return victims_per_zone


def get_Bypass_GreenInZone_Count_before(df, triage_events, victims_per_zone, timestamp):
    df = df[df['zone'] != -1]
    df_visits = df.loc[df['zone'] != df['zone'].shift()]
    count = 0
    for i in range(len(df_visits)):
        zone_id = df_visits.iloc[i]['zone']
        if zone_id not in victims_per_zone:
            continue
        visit_start = df_visits.iloc[i]['msg.timestamp']
        if i == len(df_visits) - 1:
            visit_end = df['msg.timestamp'].max()
        else:
            visit_end = df_visits.iloc[i+1]['msg.timestamp']
        zone_triage_events = triage_events[(triage_events['zone'] == zone_id) & 
                                           (triage_events['data.color'] == 'Green')]
        num_victims = victims_per_zone[zone_id]
        victims_previously_saved = len(zone_triage_events[zone_triage_events['msg.timestamp'] < visit_start])
        victims_available = num_victims - victims_previously_saved
        victims_saved = len(zone_triage_events[(zone_triage_events['msg.timestamp'] >= visit_start) & 
                                               (zone_triage_events['msg.timestamp'] <= visit_end)])
        if victims_available > 0 and victims_saved < victims_available and visit_end < timestamp:
            count += victims_available
    return count


def get_Bypass_GreenInZone_Count_after(df, triage_events, victims_per_zone, timestamp):
    df = df[df['zone'] != -1]
    df_visits = df.loc[df['zone'] != df['zone'].shift()]
    count = 0
    for i in range(len(df_visits)):
        zone_id = df_visits.iloc[i]['zone']
        if zone_id not in victims_per_zone:
            continue
        visit_start = df_visits.iloc[i]['msg.timestamp']
        if i == len(df_visits) - 1:
            visit_end = df['msg.timestamp'].max()
        else:
            visit_end = df_visits.iloc[i+1]['msg.timestamp']
        zone_triage_events = triage_events[(triage_events['zone'] == zone_id) & 
                                           (triage_events['data.color'] == 'Green')]
        num_victims = victims_per_zone[zone_id]
        victims_previously_saved = len(zone_triage_events[zone_triage_events['msg.timestamp'] < visit_start])
        victims_available = num_victims - victims_previously_saved
        victims_saved = len(zone_triage_events[(zone_triage_events['msg.timestamp'] >= visit_start) & 
                                               (zone_triage_events['msg.timestamp'] <= visit_end)])
        if victims_available > 0 and victims_saved < victims_available and visit_end > timestamp:
            count += victims_available
    return count


def get_Bypass_YellowInZone_Count_before(df, triage_events, victims_per_zone, timestamp):
    df = df[df['zone'] != -1]
    df_visits = df.loc[df['zone'] != df['zone'].shift()]
    grouped = df_visits.groupby('zone')
    count = 0
    for zone_id, zone_df in grouped:
        if zone_id not in victims_per_zone:
            continue
        zone_visits = zone_df.sort_values(by='msg.timestamp')
        zone_triage_events = triage_events[(triage_events['zone'] == zone_id) & 
                                           (triage_events['data.color'] == 'Yellow')]
        num_victims = victims_per_zone[zone_id]
        for i in range(len(zone_visits)):
            if i == len(zone_visits) - 1:
                current_visit_timestamp = zone_visits.iloc[i]['msg.timestamp']
                next_visit_timestamp = sorted(df[df['zone'] == zone_id]['msg.timestamp'])[-1]
            else:
                current_visit_timestamp = zone_visits.iloc[i]['msg.timestamp']
                next_visit_timestamp = zone_visits.iloc[i+1]['msg.timestamp']
            victims_previously_saved = len(zone_triage_events[zone_triage_events['msg.timestamp'] < current_visit_timestamp])
            victims_available = num_victims - victims_previously_saved
            victims_saved = len(zone_triage_events[(zone_triage_events['msg.timestamp'] >= current_visit_timestamp) & 
                                                   (zone_triage_events['msg.timestamp'] <= next_visit_timestamp)])
            if victims_available > 0 and victims_saved == 0 and next_visit_timestamp < timestamp:
                count += victims_available
    return count


def get_Bypass_YellowInZone_Count_after(df, triage_events, victims_per_zone, timestamp):
    df = df[df['zone'] != -1]
    df_visits = df.loc[df['zone'] != df['zone'].shift()]
    grouped = df_visits.groupby('zone')
    count = 0
    for zone_id, zone_df in grouped:
        if zone_id not in victims_per_zone:
            continue
        zone_visits = zone_df.sort_values(by='msg.timestamp')
        zone_triage_events = triage_events[(triage_events['zone'] == zone_id) & 
                                           (triage_events['data.color'] == 'Yellow')]
        num_victims = victims_per_zone[zone_id]
        for i in range(len(zone_visits)):
            if i == len(zone_visits) - 1:
                current_visit_timestamp = zone_visits.iloc[i]['msg.timestamp']
                next_visit_timestamp = sorted(df[df['zone'] == zone_id]['msg.timestamp'])[-1]
            else:
                current_visit_timestamp = zone_visits.iloc[i]['msg.timestamp']
                next_visit_timestamp = zone_visits.iloc[i+1]['msg.timestamp']
            victims_previously_saved = len(zone_triage_events[zone_triage_events['msg.timestamp'] < current_visit_timestamp])
            victims_available = num_victims - victims_previously_saved
            victims_saved = len(zone_triage_events[(zone_triage_events['msg.timestamp'] >= current_visit_timestamp) & 
                                                   (zone_triage_events['msg.timestamp'] <= next_visit_timestamp)])
            if victims_available > 0 and victims_saved == 0 and next_visit_timestamp > timestamp:
                count += victims_available
    return count


def get_revisits_zone(df, triage_events, victims, p1, p2, p3):
    df = df.sort_values(by='msg.timestamp')
    df = df[df['zone'] != -1]
    # players_num = len(set(df['data.participant_id']))
    players_num = len(set(df['data.playername']))
    if players_num == 3:
        df_1 = df[(df['data.playername'] == 'Aaronskiy1')]  # player_1
        df_2 = df[(df['data.playername'] == 'clarkie765')]  # player_2
        df_3 = df[(df['data.playername'] == 'Agent_eito')]  # player_3

        # df_1 = df[(df['data.participant_id'] == p1)]  # player_1
        # df_2 = df[(df['data.participant_id'] == p2)]  # player_2
        # df_3 = df[(df['data.participant_id'] == p3)]  # player_3

        visits_1 = df_1.loc[df_1['zone'] != df_1['zone'].shift()]
        visits_2 = df_2.loc[df_2['zone'] != df_2['zone'].shift()]
        visits_3 = df_3.loc[df_3['zone'] != df_3['zone'].shift()]
        visits = pd.concat([visits_1, visits_2, visits_3]).sort_values(by=["msg.timestamp"])
    else:
        visits = df.loc[df['zone'] != df['zone'].shift()]
    # visits['fruitful'] = 'N/A'
    visits['revisit'] = 'False'
    visits['revisits_victim-absent'] = 'False'
    visits['visits_victim-absent'] = 'False'
    visits['revisits_victims-saved'] = 'False'
    zones_visited = []
    victims_encounter = []

    for i in range(len(visits)):
        j = i+1
        zone_id = visits.iloc[i]['zone']
        num_victims = len(victims[victims['Zone'] == zone_id])
        visit_start = visits.iloc[i]['msg.timestamp']
        if i == len(visits) - 1:
            visit_end = df['msg.timestamp'].max()
        else:
            # while (visits.iloc[i]['data.participant_id'] != visits.iloc[j]['data.participant_id'])&(j != (len(visits)-1)):
            while (visits.iloc[i]['data.playername'] != visits.iloc[j]['data.playername']) & (
                        j != (len(visits) - 1)):
                j += 1
            # else:
            #     visit_end = visits.iloc[j]['msg.timestamp']
            visit_end = visits.iloc[j]['msg.timestamp']

        zone_triage_events = triage_events[(triage_events['zone'] == zone_id)]
        yellow_victims_previously_saved = len(zone_triage_events[(zone_triage_events['msg.timestamp'] < visit_start) & (zone_triage_events['data.target_block_type'].isin(['Victim Block 1','asistmod:block_victim_1']))])
        green_victims_previously_saved = len(zone_triage_events[(zone_triage_events['msg.timestamp'] < visit_start) & (zone_triage_events['data.target_block_type'].isin(['Victim Block 2', 'asistmod:block_victim_2']))])
        # if visits.iloc[i]['yellow_cleared'] == 2:
        #     victims_available = num_victims - green_victims_previously_saved - yellow_victims
        # else:
        victims_available = num_victims - green_victims_previously_saved - yellow_victims_previously_saved
        victims_saved = len(zone_triage_events[(zone_triage_events['msg.timestamp'] >= visit_start) &
                                               (zone_triage_events['msg.timestamp'] <= visit_end)])

        if zone_id not in zones_visited:
            zones_visited.append(zone_id)
            victims_encounter.append(num_victims)
            if (victims_available == 0) & (num_victims > 0):
                visits.iloc[i, visits.columns.get_loc('visits_victim-absent')] = 'True'
            continue
        visits.iloc[i, visits.columns.get_loc('revisit')] = 'True'
        yellow_victims = len(victims[(victims['Zone'] == zone_id) & (victims['VictimColor'] == 'Yellow')])

        if num_victims == 0:
            continue
        if (victims_available == 0) & (num_victims > 0):
            visits.iloc[i, visits.columns.get_loc('revisits_victim-absent')] = 'True'

        # if victims_available > 0 and victims_saved > 0:
        #     visits.iloc[i, visits.columns.get_loc('fruitful')] = 'True'
        # elif victims_available == 0:
        #     visits.iloc[i, visits.columns.get_loc('fruitful')] = 'N/A'
        # else:
        #     visits.iloc[i, visits.columns.get_loc('fruitful')] = 'False'
        if victims_saved > 0:
            visits.iloc[i, visits.columns.get_loc('revisits_victims-saved')] = 'True'
    victims_encounter_num = np.sum(victims_encounter)
    # return visits[visits['revisit'] == 'True']
    return visits, victims_encounter_num
    

def zone_revisits_fully_cleared(triage_events, revisits, victims_zone):
    grouped = revisits.groupby('zone')
    fully_cleared = []
    for zone_id, zone_revisits in grouped:
        zone_triage_timestamps = triage_events[triage_events['zone'] == zone_id].sort_values(by='msg.timestamp')['msg.timestamp']
        zone_revisit_timestamps = zone_revisits[['msg.timestamp', 'segment', 'zone_type']]
        num_victims = victims_zone[zone_id] if zone_id in victims_zone.keys() else 0
        saved_victims = len(zone_triage_timestamps)
        if num_victims == 0 or saved_victims < num_victims:
            continue
        for zone_revisit_timestamp in zone_revisit_timestamps.values.tolist():
            revisit_timestamp, segment, zone_type = zone_revisit_timestamp
            if revisit_timestamp > zone_triage_timestamps.values[-1]:
                fully_cleared.append([revisit_timestamp, zone_id, zone_type, segment])
    df_fully_cleared = pd.DataFrame.from_records(fully_cleared, columns=["msg.timestamp", "zone", "zone_type", "segment"])
    return df_fully_cleared


def extract_bypass_variables(df, building, tag):
    try:
        victims = pd.read_csv(building.victims_file)
        victims_per_zone = get_victims_per_zone(victims)
        green_victims_per_zone = get_green_victims_per_zone(victims)
        yellow_victims_per_zone = get_yellow_victims_per_zone(victims)
        building_zones = pd.read_csv(building.zones_file)
        zone_coords = building_zones[['Zone Number', 'Zone Type', 'Xcoords-TopLeft', 'XCoords-BotRight',
                                    'Zcoords-TopLeft', 'ZCoords-BotRight']].values.tolist()
        df = df[df['data.mission_timer'] != "Mission Timer not initialized."]
        df = df[df['data.mission_timer'].notna()]        
        
        half_time = df[df['msg.sub_type'] == 'Event:VictimsExpired']['msg.timestamp'].values[0]
        df['segment'] = df.apply(lambda row: 1 if row['msg.timestamp'] <  half_time else 2, axis=1)  
        
        df['zone'] = df.apply(lambda row: assign_zone(row['data.x'], row['data.z'], zone_coords), axis=1)
        df['zone'] = df['zone'].astype(int)
        
        triage_events = df[(df['msg.sub_type'] == 'Event:Triage') & (df['data.triage_state'] == 'SUCCESSFUL')]
        triage_events = triage_events.drop_duplicates(subset=['data.victim_x', 'data.victim_z'], keep='last')
        
        triage_events['zone'] = triage_events.apply(lambda row: assign_victim_zone(row['data.victim_x'], row['data.victim_z'], victims), axis=1)
        triage_events['zone'] = triage_events['zone'].astype(int)
       
        num_yellows = 10
        yellow_triages = triage_events[triage_events['data.color'] == 'Yellow']
        if len(yellow_triages) == num_yellows:
            all_yellow_cleared = yellow_triages['msg.timestamp'].max()
        else:
            all_yellow_cleared = df[df['segment'] == 2]['msg.timestamp'].min()
                
        bypass_green_in_zone_count_before_yellowcleared = get_Bypass_GreenInZone_Count_before(df, triage_events, green_victims_per_zone, all_yellow_cleared)
        bypass_green_in_zone_count_after_yellowcleared = get_Bypass_GreenInZone_Count_after(df, triage_events, green_victims_per_zone, all_yellow_cleared)
        bypass_yellow_in_zone_count_before_yellowcleared = get_Bypass_YellowInZone_Count_before(df, triage_events, yellow_victims_per_zone, all_yellow_cleared)

        all_victims_visited = df.iloc[[0]]['msg.timestamp'].values[0]
        zones_with_victims = list(victims_per_zone.keys())
        for _, row in df.iterrows(): 
            if len(zones_with_victims) == 0:
                break
            zone = row['zone'] 
            if zone in zones_with_victims:
                zones_with_victims.remove(zone)
                all_victims_visited = row['msg.timestamp']
        if len(zones_with_victims) == 0:
            bypass_green_in_zone_count_before_victimsfound = get_Bypass_GreenInZone_Count_before(df, triage_events, green_victims_per_zone, all_victims_visited)
            bypass_green_in_zone_count_after_victimsfound = get_Bypass_GreenInZone_Count_after(df, triage_events, green_victims_per_zone, all_victims_visited)
            bypass_yellow_in_zone_count_before_victimsfound = get_Bypass_YellowInZone_Count_before(df, triage_events, yellow_victims_per_zone, all_victims_visited)
            bypass_yellow_in_zone_count_after_victimsfound = get_Bypass_YellowInZone_Count_after(df, triage_events, yellow_victims_per_zone, all_victims_visited)
        else:
            bypass_green_in_zone_count_before_victimsfound = -99
            bypass_green_in_zone_count_after_victimsfound = -99
            bypass_yellow_in_zone_count_before_victimsfound = -99
            bypass_yellow_in_zone_count_after_victimsfound = -99
    except:
        bypass_yellow_in_zone_count_before_yellowcleared = -99
        bypass_green_in_zone_count_before_yellowcleared = -99
        bypass_green_in_zone_count_after_yellowcleared = -99
        bypass_green_in_zone_count_before_victimsfound = -99
        bypass_green_in_zone_count_after_victimsfound = -99
        bypass_yellow_in_zone_count_before_victimsfound = -99
        bypass_yellow_in_zone_count_after_victimsfound = -99
    bypass_counts_df = pd.DataFrame.from_records([[bypass_green_in_zone_count_before_yellowcleared,
                                                   bypass_green_in_zone_count_after_yellowcleared,
                                                   bypass_yellow_in_zone_count_before_yellowcleared,
                                                   bypass_green_in_zone_count_before_victimsfound,
                                                   bypass_green_in_zone_count_after_victimsfound,
                                                   bypass_yellow_in_zone_count_before_victimsfound,
                                                   bypass_yellow_in_zone_count_after_victimsfound
                                                   ]], 
                                         columns=[tag + "BypassedGreenInZone_Count_Before_All_Yellows_Cleared",
                                                  tag + "BypassedGreenInZone_Count_After_All_Yellows_Cleared",
                                                  tag + "BypassedYellowInZone_Count_Before_All_Yellows_Cleared",
                                                  tag + "BypassedGreenInZone_Count_Before_All_Victims_Found",
                                                  tag + "BypassedGreenInZone_Count_After_All_Victims_Found",
                                                  tag + "BypassedYellowInZone_Count_Before_All_Victims_Found",
                                                  tag + "BypassedYellowInZone_Count_After_All_Victims_Found"
                                                  ])
    return bypass_counts_df


def extract_revisit_variables(df, building, tag):
    try:
        victims = pd.read_csv(building.victims_file)
        victims_per_zone = get_victims_per_zone(victims)
        building_zones = pd.read_csv(building.zones_file)
        zone_coords = building_zones[['Zone Number', 'Zone Type', 'Xcoords-TopLeft', 'XCoords-BotRight',
                                    'Zcoords-TopLeft', 'ZCoords-BotRight']].values.tolist()
        zone_types = building_zones[['Zone Number', 'Zone Type']].values.tolist()
        zone_mapping = {}
        for entry in zone_types:
            zone_mapping[entry[0]] = entry[1]
        zone_mapping[-1] = -1
        df = df[df['data.mission_timer'] != "Mission Timer not initialized."]
        df = df[df['data.mission_timer'].notna()]        
        half_time = df[df['msg.sub_type'] == 'Event:VictimsExpired']['msg.timestamp'].values[0]
        df['segment'] = df.apply(lambda row: 1 if row['msg.timestamp'] <  half_time else 2, axis=1)  
        
        df['zone'] = df.apply(lambda row: assign_zone(row['data.x'], row['data.z'], zone_coords), axis=1)
        df['zone'] = df['zone'].astype(int)
        df['zone_type'] = df.apply(lambda row: zone_mapping[row['zone']], axis=1)
        
        triage_events = df[(df['msg.sub_type'] == 'Event:Triage') & (df['data.triage_state'] == 'SUCCESSFUL')]
        triage_events = triage_events.drop_duplicates(subset=['data.victim_x', 'data.victim_z'], keep='last')
        
        triage_events['zone'] = triage_events.apply(lambda row: assign_victim_zone(row['data.victim_x'], row['data.victim_z'], victims), axis=1)
        triage_events['zone'] = triage_events['zone'].astype(int)
        
        num_yellows = 10
        yellow_triages = triage_events[triage_events['data.color'] == 'Yellow']
        if len(yellow_triages) == num_yellows:
            all_yellow_cleared = yellow_triages.iloc[[-1]]['msg.timestamp'].values[0]
        else:
            all_yellow_cleared = df[df['segment'] == 2]['msg.timestamp'].min()
        df['yellow_cleared'] = df.apply(lambda row: 1 if row['msg.timestamp'] < all_yellow_cleared else 2, axis=1)
        
        victims_found = df.iloc[[0]]['msg.timestamp'].values[0]
        zones_with_victims = list(victims_per_zone.keys())
        for _, row in df.iterrows(): 
            if len(zones_with_victims) == 0:
                break
            zone = row['zone'] 
            if zone in zones_with_victims:
                zones_with_victims.remove(zone)
                victims_found = row['msg.timestamp']
        if len(zones_with_victims) != 0:
            victims_found = df['msg.timestamp'].max()

        df_revisits = get_revisits_zone(df, triage_events, victims)
        df_fully_cleared = zone_revisits_fully_cleared(triage_events, df_revisits, victims_per_zone)    
        df_fully_cleared['yellow_cleared'] = df_fully_cleared.apply(lambda row: 1 if row['msg.timestamp'] < all_yellow_cleared else 2, axis=1)
        df_revisits['victims_found'] = df_revisits.apply(lambda row: 1 if row['msg.timestamp'] < victims_found else 2, axis=1)

        revisits_room_fruitful_before_yellowcleared = len(df_revisits[(df_revisits['fruitful'] == 'True') & (df_revisits['yellow_cleared'] == 1) & (df_revisits['zone_type'] == 3)])
        revisits_room_fruitful_after_yellowcleared = len(df_revisits[(df_revisits['fruitful'] == 'True') & (df_revisits['yellow_cleared'] == 2) & (df_revisits['zone_type'] == 3)])

        revisits_room_fruitful_ratio_after_yellowcleared_to_before = float(revisits_room_fruitful_after_yellowcleared/revisits_room_fruitful_before_yellowcleared) if revisits_room_fruitful_before_yellowcleared != 0 else str(revisits_room_fruitful_after_yellowcleared) + ':' + str(revisits_room_fruitful_before_yellowcleared)

        revisits_room_fruitful_before_victimsfound = len(df_revisits[(df_revisits['fruitful'] == 'True') & (df_revisits['victims_found'] == 1) & (df_revisits['zone_type'] == 3)])
        revisits_room_fruitful_after_victimsfound = len(df_revisits[(df_revisits['fruitful'] == 'True') & (df_revisits['victims_found'] == 2) & (df_revisits['zone_type'] == 3)])

        ratio_revisits_room_fruitful_after_victimsfound_to_before = float(revisits_room_fruitful_after_victimsfound/revisits_room_fruitful_before_victimsfound) if revisits_room_fruitful_before_victimsfound != 0 else str(revisits_room_fruitful_after_victimsfound) + ':' + str(revisits_room_fruitful_before_victimsfound)

        revisits_room_notfruitful_before_yellowcleared = len(df_revisits[(df_revisits['fruitful'] == 'False') & (df_revisits['zone_type'] == 3) & (df_revisits['yellow_cleared'] == 1)])
        revisits_room_notfruitful_after_yellowcleared = len(df_revisits[(df_revisits['fruitful'] == 'False') & (df_revisits['zone_type'] == 3) & (df_revisits['yellow_cleared'] == 2)])

        revisits_fully_cleared_room_before_yellowcleared = len(df_fully_cleared[(df_fully_cleared['zone_type'] == 3) & (df_fully_cleared['yellow_cleared'] == 1)])
        revisits_fully_cleared_room_after_yellowcleared = len(df_fully_cleared[(df_fully_cleared['zone_type'] == 3) & (df_fully_cleared['yellow_cleared'] == 2)])
        
        revisits_fully_cleared_room = len(df_fully_cleared[df_fully_cleared['zone_type'] == 3])
        revisits_room_notfruitful = len(df_revisits[(df_revisits['fruitful'] == 'False') & (df_revisits['zone_type'] == 3)])

        zones_in_building = building_zones['Zone Number'].unique()
        zones_with_victims = list(victims_per_zone.keys())
        zones_revisited = [zone for zone in zones_in_building if zone in df_revisits['zone'].unique()]
        zones_revisited_before_halftime = [zone for zone in zones_in_building if zone in df_revisits[df_revisits['segment'] == 1]['zone'].unique()]
        zones_revisited_after_halftime = [zone for zone in zones_in_building if zone in df_revisits[df_revisits['segment'] == 2]['zone'].unique()]
        zones_revisited_with_victims = [zone for zone in zones_with_victims if zone in df_revisits['zone'].unique()]

        percent_zones_revisited = round(float(len(zones_revisited)/len(zones_in_building)) * 100, 2)
        percent_zones_revisited_before_halftime = round(float(len(zones_revisited_before_halftime)/len(zones_in_building)) * 100, 2)
        percent_zones_revisited_after_halftime = round(float(len(zones_revisited_after_halftime)/len(zones_in_building)) * 100, 2)
        percent_zones_revisited_with_victims = round(float(len(zones_revisited_with_victims)/len(zones_with_victims)) * 100, 2)

    except:
       revisits_room_fruitful_before_yellowcleared = -99
       revisits_room_fruitful_after_yellowcleared = -99
       revisits_room_fruitful_ratio_after_yellowcleared_to_before = -99
       revisits_room_fruitful_before_victimsfound = -99
       revisits_room_fruitful_after_victimsfound = -99
       revisits_room_notfruitful_before_yellowcleared = -99
       revisits_room_notfruitful_after_yellowcleared = -99
       revisits_fully_cleared_room_before_yellowcleared = -99
       revisits_fully_cleared_room_after_yellowcleared = -99
       revisits_room_notfruitful = -99
       revisits_fully_cleared_room = -99
       ratio_revisits_room_fruitful_after_victimsfound_to_before = -99
       percent_zones_revisited = -99
       percent_zones_revisited_before_halftime = -99
       percent_zones_revisited_after_halftime = -99
       percent_zones_revisited_with_victims = -99
    df_revisits = pd.DataFrame.from_records([[revisits_room_fruitful_after_yellowcleared,
                                              revisits_room_fruitful_ratio_after_yellowcleared_to_before, 
                                              revisits_room_notfruitful_before_yellowcleared,
                                              revisits_room_notfruitful_after_yellowcleared,
                                              revisits_fully_cleared_room_before_yellowcleared,
                                              revisits_fully_cleared_room_after_yellowcleared,
                                              revisits_room_notfruitful,
                                              revisits_fully_cleared_room,
                                              ratio_revisits_room_fruitful_after_victimsfound_to_before,
                                              percent_zones_revisited,
                                              percent_zones_revisited_before_halftime,
                                              percent_zones_revisited_after_halftime,
                                              percent_zones_revisited_with_victims             
                                             ]],
                                            columns=[tag + "Revisits_Rooms_Fruitful_After_All_Yellows_Cleared",
                                                     tag + "Revisits_Rooms_Fruitful_Ratio_After_All_Yellows_Cleared_to_Before",
                                                     tag + "Revisits_Rooms_NotFruitful_Before_All_Yellows_Cleared",
                                                     tag + "Revisits_Rooms_NotFruitful_After_All_Yellows_Cleared",
                                                     tag + "Revisits_fully_cleared_Rooms_Before_All_Yellows_Cleared",
                                                     tag + "Revisits_fully_cleared_Rooms_After_All_Yellows_Cleared",
                                                     tag + "Revisits_Rooms_NotFruitful",
                                                     tag + "Revisits_fully_cleared_Rooms",
                                                     tag + "Ratio_Revisits_Rooms_Fruitful_After_All_Victims_Found_to_Before",
                                                     tag + "Percent_Zones_Revisited",
                                                     tag + "Percent_Zones_Revisited_Before_All_Yellows_Cleared",
                                                     tag + "Percent_Zones_Revisited_After_All_Yellows_Cleared",
                                                     tag + "Percent_Zones_Revisited_With_Victims"
                                                    ])                                              
    return df_revisits


def get_victims_encounter(df, df_visit, victims_in_zone, triage_events, picked_event):
    victims_encounter = []
    for i in range(len(df_visit)):
        df_pre = df[(df['msg.timestamp'] < df_visit.iloc[i]['msg.timestamp'])]
        df_pre_picked = picked_event[picked_event['msg.timestamp'] < df_visit.iloc[i]['msg.timestamp']]
        victims_pre_pick = df_pre_picked.drop_duplicates(subset=['data.victim_id'], keep='last')
        df_pre_saved = triage_events[triage_events['msg.timestamp'] < df_visit.iloc[i]['msg.timestamp']]
        if len(victims_in_zone['Metadata_ID']) ==0 and len(df_pre_picked)==0:
            continue
        else:
            vic_pre = df_pre[~df_pre['data.victim_id'].isna()]
            # static_vic = df_pre[(df_pre['data.victim_id'].isin(set(victims_in_zone['Metadata_ID'])))&(~(df_pre['data.victim_id'].isin(victims['data.victim_id'])))&(~(df_pre['data.victim_id'].isin(set(df_pre_saved['data.victim_id']))))]
            # static_vic = df_pre[(~(df_pre['data.victim_id'].isin(victims['data.victim_id']))) & (
            #                         ~(df_pre['data.victim_id'].isin(set(df_pre_saved['data.victim_id']))))]
            if len(vic_pre) == 0:
                static_vic = victims_in_zone
            else:
                static_vic = victims_in_zone[(~(victims_in_zone['Metadata_ID'].isin(victims_pre_pick['data.victim_id']))) & (
                                    ~(victims_in_zone['Metadata_ID'].isin(set(df_pre_saved['data.victim_id']))))]

                victims_ori = victims_pre_pick[victims_pre_pick['data.victim_id'].isin(set(victims_in_zone['Metadata_ID']))]
                victims_new = victims_pre_pick[~victims_pre_pick['data.victim_id'].isin(set(victims_in_zone['Metadata_ID']))]
                picked = []
                for j in range(len(victims_ori)):
                    # vic = victims_pre_pick.iloc[j]
                    vic = victims_ori.iloc[j]
                    if vic['msg.sub_type'] == 'Event:VictimPickedUp':
                        picked_victim_id = vic['data.victim_id']
                        picked.append(picked_victim_id)
                        continue
                    else:
                        proximity = np.sqrt(np.diff([vic['data.x'], df_visit.iloc[i]['data.x']]) ** 2 + np.diff([vic['data.z'], df_visit.iloc[i]['data.z']]) ** 2)
                        if proximity <=3:
                            victims_encounter.append(vic['data.victim_id'])
                for k in range(len(victims_new)):
                    vic = victims_new.iloc[k]
                    proximity = np.sqrt(np.diff([vic['data.x'], df_visit.iloc[i]['data.x']]) ** 2 + np.diff([vic['data.z'], df_visit.iloc[i]['data.z']]) ** 2)
                    if proximity <= 3:
                        victims_encounter.append(vic['data.victim_id'])
                    else:
                        continue
            for m in range(len(static_vic)):
                vic = static_vic.iloc[m]
                proximity = np.sqrt(np.diff([vic['X_coord'], df_visit.iloc[i]['data.x']]) ** 2 + np.diff([vic['Z_coord'], df_visit.iloc[i]['data.z']]) ** 2)
                if proximity <= 3:
                    victims_encounter.append(vic['Metadata_ID'])
                else:
                    continue
    return len(set(victims_encounter))


def get_zones_visited_with_victims(df, triage_events, picked_event, victims, p1, p2, p3):
    df = df.sort_values(by='msg.timestamp')
    df = df[df['zone'] != -1]
    # players_num = len(set(df['data.participant_id']))
    players_num = len(set(df['data.playername']))
    if players_num == 3:
        # df_1 = df[(df['data.playername'] == 'Aaronskiy1')]  # player_1
        # df_2 = df[(df['data.playername'] == 'clarkie765')]  # player_2
        # df_3 = df[(df['data.playername'] == 'Agent_eito')]  # player_3

        # df_1 = df[(df['data.participant_id'] == p1)]  # player_1
        # df_2 = df[(df['data.participant_id'] == p2)]  # player_2
        # df_3 = df[(df['data.participant_id'] == p3)]  # player_3

        df_1 = df[(df['data.playername'] == p1)]  # player_1
        df_2 = df[(df['data.playername'] == p2)]  # player_2
        df_3 = df[(df['data.playername'] == p3)]  # player_3

        visits_1 = df_1.loc[df_1['zone'] != df_1['zone'].shift()]
        visits_2 = df_2.loc[df_2['zone'] != df_2['zone'].shift()]
        visits_3 = df_3.loc[df_3['zone'] != df_3['zone'].shift()]
        visits = pd.concat([visits_1, visits_2, visits_3]).sort_values(by=["msg.timestamp"])
    else:
        visits = df.loc[df['zone'] != df['zone'].shift()]
    # visits['fruitful'] = 'N/A'

    victims_per_zone_list = list(victims.groupby("Zone"))

    visits['visits_victim-present'] = 'False'
    visits['visits_victim-absent'] = 'False'
    visits['revisit'] = 'False'
    visits['revisits_victim-present'] = 'False'
    visits['revisits_victim-absent'] = 'False'
    visits['revisits_victims-saved'] = 'False'
    visits['visit_redundant'] = 'False'
    zones_visited = []
    victims_encounter = []
    zones_visitors = defaultdict(set)
    zone_first_visitor = {}
    for i in range(len(visits)):
        j = i + 1
        zone_id = visits.iloc[i]['zone']
        victims_in_zone = victims[victims['Zone'] == zone_id]
        num_victims = len(victims_in_zone)
        visit_start = visits.iloc[i]['msg.timestamp']
        # player = visits.iloc[i]['data.participant_id']
        player = visits.iloc[i]['data.playername']
        if i == len(visits) - 1:
            visit_end = df['msg.timestamp'].max()
        else:
            # while (visits.iloc[i]['data.participant_id'] != visits.iloc[j]['data.participant_id']) & (j != (len(visits) - 1)):
            while (visits.iloc[i]['data.playername'] != visits.iloc[j]['data.playername']) & (j != (len(visits) - 1)):
                j += 1
            # else:
            #     visit_end = visits.iloc[j]['msg.timestamp']
            visit_end = visits.iloc[j]['msg.timestamp']

        # df_visit = df[(df['msg.timestamp'] >= visit_start) & (df['msg.timestamp'] < visit_end) & (df['data.participant_id']==player)]
        # victims_encounter_visit = get_victims_encounter(df, df_visit, victims_in_zone, triage_events, picked_event)

        if len(picked_event) > 0:
            victim_picked = picked_event[picked_event['msg.sub_type'] == 'Event:VictimPickedUp']
            victim_placed = picked_event[picked_event['msg.sub_type'] == 'Event:VictimPlaced']
            zone_victim_picked = victim_picked[(victim_picked['zone'] == zone_id)]
            zone_victim_placed = victim_placed[(victim_placed['zone'] == zone_id)]

            victim_picked_previously = zone_victim_picked[(zone_victim_picked['msg.timestamp'] < visit_start)]
            victim_placed_previously = zone_victim_placed[(zone_victim_placed['msg.timestamp'] < visit_start)]

            victim_picked_previously_len = len(victim_picked_previously)
            victim_placed_previously_len = len(victim_placed_previously)

            # victim_moved = pd.concat([victim_picked_previously, victim_placed_previously]).sort_values(by=["msg.timestamp"])
            # victim_moved = victim_moved.drop_duplicates(subset=['data.victim_id'], keep='last')


        else:
            victim_picked_previously_len = 0
            victim_placed_previously_len = 0
        if len(triage_events)>0:
            zone_triage_events = triage_events[(triage_events['zone'] == zone_id)]
            yellow_victims_previously_saved = len(zone_triage_events[(zone_triage_events['msg.timestamp'] < visit_start) & (zone_triage_events['data.target_block_type'].isin(['Victim Block 1','asistmod:block_victim_1']))])
            green_victims_previously_saved = len(zone_triage_events[(zone_triage_events['msg.timestamp'] < visit_start) & (zone_triage_events['data.target_block_type'].isin(['Victim Block 2', 'asistmod:block_victim_2']))])

            victims_saved = len(zone_triage_events[(zone_triage_events['msg.timestamp'] >= visit_start) &
                                                   (zone_triage_events['msg.timestamp'] < visit_end)])
        else:
            yellow_victims_previously_saved = 0
            green_victims_previously_saved = 0
            victims_saved = 0
        victims_available = num_victims - victim_picked_previously_len + victim_placed_previously_len - yellow_victims_previously_saved - green_victims_previously_saved

        if zone_id not in zones_visited:
            zones_visited.append(zone_id)
            victims_encounter.append(victims_available)
            # victims_encounter.append(victims_encounter_visit)

            # zone_first_visitor[zone_id] = visits.iloc[i]['data.participant_id']
            # zones_visitors[zone_id].add(visits.iloc[i]['data.participant_id'])
            zone_first_visitor[zone_id] = visits.iloc[i]['data.playername']
            zones_visitors[zone_id].add(visits.iloc[i]['data.playername'])
            if victims_available > 0:
                visits.iloc[i, visits.columns.get_loc('visits_victim-present')] = 'True'
            elif (victims_available == 0) & (num_victims > 0):
                visits.iloc[i, visits.columns.get_loc('visits_victim-absent')] = 'True'

            continue
        else:
            # zones_visitors[zone_id].add(visits.iloc[i]['data.participant_id'])
            zones_visitors[zone_id].add(visits.iloc[i]['data.playername'])
            # if visits.iloc[i]['data.participant_id'] != zone_first_visitor[zone_id]:
            if visits.iloc[i]['data.playername'] != zone_first_visitor[zone_id]:
                visits.iloc[i, visits.columns.get_loc('visit_redundant')] = 'True'
            elif visits.iloc[i]['data.playername'] in set(zones_visitors[zone_id]):
                visits.iloc[i, visits.columns.get_loc('revisit')] = 'True'

            # yellow_victims = len(victims[(victims['Zone'] == zone_id) & (victims['VictimColor'] == 'Yellow')])
            #
            # if num_victims == 0:
            #     continue
            if victims_available > 0:
                visits.iloc[i, visits.columns.get_loc('revisits_victim-present')] = 'True'
            elif (victims_available == 0) & (num_victims > 0):
                visits.iloc[i, visits.columns.get_loc('revisits_victim-absent')] = 'True'

            # if victims_available > 0 and victims_saved > 0:
            #     visits.iloc[i, visits.columns.get_loc('fruitful')] = 'True'
            # elif victims_available == 0:
            #     visits.iloc[i, visits.columns.get_loc('fruitful')] = 'N/A'
            # else:
            #     visits.iloc[i, visits.columns.get_loc('fruitful')] = 'False'
            if victims_saved > 0:
                visits.iloc[i, visits.columns.get_loc('revisits_victims-saved')] = 'True'
    victims_encounter_num = np.sum(victims_encounter)
    # return visits[visits['revisit'] == 'True']
    return visits, victims_encounter_num


def extract_visit_variables(df, building_zones, victims, p1, p2, p3, tag):
    try:
        zone_coords = building_zones[['ZonesDesignation', 'XCoordTopLeft_base', 'XCoordBotRight_base',
                                    'ZCoordTopLeft_base', 'ZCoordBotRight_base']].values.tolist()
        victims['Zone'] = victims.apply(lambda row: assign_zone(row['X_coord'], row['Z_coord'], zone_coords, padding=0.5), axis=1)
        victims_per_zone = get_victims_per_zone(victims)
        # zone_types = building_zones[['Zone Number', 'Zone Type']].values.tolist()
        # zone_mapping = {}
        # for entry in zone_types:
        #     zone_mapping[entry[0]] = entry[1]
        # zone_mapping[-1] = -1

        df = df[df['data.mission_timer'] != "Mission Timer not initialized."]
        df = df[df['data.mission_timer'].notna()]
        high_risk_expire_time = pd.to_datetime(df['msg.timestamp'].min()) + timedelta(minutes=2)

        df['zone'] = df.apply(lambda row: assign_zone(row['data.x'],row['data.z'], zone_coords, padding=0.5), axis=1)
        df['zone'] = df['zone'].astype(int)

        df_picked = df[(df['msg.sub_type'].isin(['Event:VictimPickedUp', 'Event:VictimPlaced']))]
        if len(df_picked)>0:
            df_picked['zone'] = df_picked.apply(lambda row: assign_zone(row['data.victim_x'], row['data.victim_z'], zone_coords, padding=0.5), axis=1)
            df_picked['zone'] = df_picked['zone'].astype(int)
        # df_victim_picked = df[df['msg.sub_type'] == 'Event:VictimPickedUp']
        # df_victim_placed = df[df['msg.sub_type']=='Event:VictimPlaced']

        medical_triages = df[(df['data.tool_type'].isin(['medicalkit', 'MEDKIT']))]
        medical_triages = medical_triages.drop_duplicates(subset=['data.target_block_x', 'data.target_block_z'], keep='last')
        # medical_triages['zone'] = medical_triages.apply(lambda row: assign_victim_zone(row['data.target_block_z'], row['data.target_block_x'], victims), axis=1)
        if len(medical_triages)>0:
            medical_triages['zone'] = medical_triages.apply(lambda row: assign_zone(row['data.target_block_x'], row['data.target_block_z'], zone_coords), axis=1)
            medical_triages['zone'] = medical_triages['zone'].astype(int)

        # half_time = df[df['msg.sub_type'] == 'Event:VictimsExpired']['msg.timestamp'].values[0]
        # df['segment'] = df.apply(lambda row: 1 if row['msg.timestamp'] <  half_time else 2, axis=1)
        
        zones_in_building = building_zones['ZonesDesignation'].unique()
        zones_with_victims = list(victims_per_zone.keys())
        zones_visited = [zone for zone in zones_in_building if zone in df['zone'].unique()]

        zones_visited_with_victims = [zone for zone in zones_with_victims if zone in df['zone'].unique()]
        # zones_visited_before_halftime = [zone for zone in zones_in_building if zone in df[df['segment'] == 1]['zone'].unique()]
        # zones_visited_after_halftime = [zone for zone in zones_in_building if zone in df[df['segment'] == 2]['zone'].unique()]

        df = pd.concat([df, medical_triages, df_picked]).sort_values(by=["msg.timestamp"])
        df_z_visits, victims_encounter = get_zones_visited_with_victims(df, medical_triages, df_picked, victims, p1, p2, p3)
        # df_z_visits, victims_encounter = get_revisits_zone(df, medical_triages, victims, p1, p2, p3)
        df_z_revisits = df_z_visits[df_z_visits['revisit'] == 'True']
        # df_revisits['victims_found'] = df_revisits.apply(lambda row: 1 if row['msg.timestamp'] < victims_found else 2, axis=1)
        df_visits_victim_present = df_z_visits[df_z_visits['visits_victim-present'] == 'True']
        df_visits_victim_absent = df_z_visits[df_z_visits['visits_victim-absent'] == 'True']
        df_revisits_victim_present = df_z_visits[df_z_visits['revisits_victim-present'] == 'True']
        df_revisits_victim_absent = df_z_visits[df_z_visits['revisits_victim-absent'] == 'True']
        df_revisits_victims_saved = df_z_visits[df_z_visits['revisits_victims-saved'] == 'True']

        df_redundant_visits = df_z_visits[df_z_visits['visit_redundant'] == 'True']

        visits_victim_present = [zone for zone in zones_with_victims if zone in df_visits_victim_present['zone'].unique()]
        visits_victim_absent = [zone for zone in zones_with_victims if zone in df_visits_victim_absent['zone'].unique()]
        zones_revisited = [zone for zone in zones_in_building if zone in df_z_revisits['zone'].unique()]
        zones_revisited_victim_present = [zone for zone in zones_with_victims if zone in df_revisits_victim_present['zone'].unique()]
        revisits_victim_absent = [zone for zone in zones_with_victims if zone in df_revisits_victim_absent['zone'].unique()]
        revisits_victims_saved = [zone for zone in zones_with_victims if zone in df_revisits_victims_saved['zone'].unique()]
        zones_redundant_visits = [zone for zone in zones_in_building if zone in df_redundant_visits['zone'].unique()]

        # percent_zones_visited = round(float(len(zones_visited)/len(zones_in_building)) * 100, 2)
        # percent_zones_visited_before_halftime = round(float(len(zones_visited_before_halftime)/len(zones_in_building)) * 100, 2)
        # percent_zones_visited_after_halftime = round(float(len(zones_visited_after_halftime)/len(zones_in_building)) * 100, 2)
        # percent_zones_visited_with_victims = round(float(len(zones_visited_with_victims)/len(zones_with_victims)) * 100, 2)

        # rooms_with_victims = [zone for zone in list(victims_per_zone.keys()) if zone_mapping[zone] == 3]
        # rooms_visited = [zone for zone in df[df['zone_type'] == 3]['zone'].unique() if zone in rooms_with_victims]
        # rooms_with_victims_visited = len(rooms_visited)

        zones_visited_num = len(zones_visited)
        zones_visited_with_victims_num = len(visits_victim_present)
        visits_victim_absent_num = len(visits_victim_absent)
        zones_revisited_num = len(zones_revisited)
        zones_revisited_with_victims_num = len(zones_revisited_victim_present)
        revisits_victim_absent_num = len(revisits_victim_absent)
        revisits_victims_saved_num = len(revisits_victims_saved)
        zones_redundant_visits_num = len(zones_redundant_visits)

    except:
        zones_visited_num = -99
        zones_visited_with_victims_num = -99
        visits_victim_absent_num = -99
        zones_revisited_num = -99
        zones_revisited_with_victims_num = -99
        revisits_victim_absent_num = -99
        revisits_victims_saved_num = -99
        victims_encounter = -99
        zones_redundant_visits_num = -99

    df_visits = pd.DataFrame.from_records([[zones_visited_num,
                                            zones_visited_with_victims_num,
                                            visits_victim_absent_num,
                                            zones_revisited_num,
                                            zones_redundant_visits_num,
                                            zones_revisited_with_victims_num,
                                            revisits_victim_absent_num,
                                            revisits_victims_saved_num,
                                            victims_encounter
                                           ]],
                                          columns=["Zones_visits",
                                                   "VictimZones_visits_victim-present",
                                                   "VictimZones_visits_victim-absent",
                                                   "Zones_revisits",
                                                   'Zones_redundant_visits',
                                                   "VictimZones_revisits_victim-present",
                                                   "VictimZones_revisits_victim-absent",
                                                   "VictimZones_revisits_victims-saved",
                                                   "Victim_encounters"
                                                   ])
    return df_visits


