import pandas as pd
import numpy as np


def compute_distance(df):
    df = df.drop_duplicates('msg.timestamp', keep='first')
    df['d'] = None
    for i in range(len(df)):
        if i == 0:
            df['d'].iloc[i] = 0
            continue
        else:
            df['d'].iloc[i] = np.sqrt(np.diff([df['data.x'].iloc[i], df['data.x'].iloc[i-1]]) ** 2 + np.diff([df['data.z'].iloc[i], df['data.z'].iloc[i-1]]) ** 2)
    return df['d'].sum()[0]


def unfreeze_count(df, p1, p2, p3):
    df_ft = df[df['msg.sub_type'] == 'Event:PlayerFrozenStateChange']
    if len(df_ft) > 0:
        df_unfreeze = df_ft[df_ft['data.state_changed_to'] == 'UNFROZEN']
        # df1_unfreeze = df_unfreeze[df_unfreeze['data.medic_participant_id'] == p1]
        # df2_unfreeze = df_unfreeze[df_unfreeze['data.medic_participant_id'] == p2]
        # df3_unfreeze = df_unfreeze[df_unfreeze['data.medic_participant_id'] == p3]

        df1_unfreeze = df_unfreeze[df_unfreeze['data.medic_playername'] == p1]
        df2_unfreeze = df_unfreeze[df_unfreeze['data.medic_playername'] == p2]
        df3_unfreeze = df_unfreeze[df_unfreeze['data.medic_playername'] == p3]

        unfreeze_num_p1 = len(df1_unfreeze)
        unfreeze_num_p2 = len(df2_unfreeze)
        unfreeze_num_p3 = len(df3_unfreeze)
    else:
        unfreeze_num_p1 = 0
        unfreeze_num_p2 = 0
        unfreeze_num_p3 = 0
    unfreeze_p1_df = pd.DataFrame.from_records([[unfreeze_num_p1]],
                                             columns=['Unfreeze_teammate_count'])
    unfreeze_p2_df = pd.DataFrame.from_records([[unfreeze_num_p2]],
                                             columns=['Unfreeze_teammate_count'])
    unfreeze_p3_df = pd.DataFrame.from_records([[unfreeze_num_p3]],
                                             columns=['Unfreeze_teammate_count'])

    return unfreeze_p1_df, unfreeze_p2_df, unfreeze_p3_df


def victim_classification(victim_triaged_id, victim_classes):
    # for j in range(len(victim_classes)):
    #     if victim_triaged_id == victim_classes.iloc[j]['Metadata_ID']:
    #         victim_class = victim_classes.iloc[j]['Class']
    #         break
    victim_class = 0
    victim_class = victim_classes[victim_classes['Metadata_ID'] == victim_triaged_id]['Class'].item()
    # if victim_class == None:
    #     victim_class = 0
    return victim_class


def rubble_classification(rubble_coord_x,rubble_coord_z, rubble_classes):
    rubble_class = 0
    for j in range(len(rubble_classes)):
        if (rubble_coord_x == rubble_classes.iloc[j]['X_coord']) and (rubble_coord_z == rubble_classes.iloc[j]['Y_coord']):
            rubble_class = rubble_classes.iloc[j]['Classification']
    # if rubble_class == None:
    #     rubble_class = 0
    return rubble_class


def extract_triage_variables(df, victim_classes, rubble_classes, tag):
    try:
        # medical_triages = df[(df['data.tool_type'].isin(['medicalkit', 'MEDKIT']))]
        # medical_triages = medical_triages.drop_duplicates(subset=['data.target_block_x', 'data.target_block_z'], keep='last')

        medical_triages = df[(df['msg.sub_type'] == 'Event:Triage') & (df['data.triage_state'] == 'SUCCESSFUL')]
        medical_triages = medical_triages.drop_duplicates(subset=['data.victim_x', 'data.victim_z'], keep='last')
        victims_low_value = medical_triages[(medical_triages['data.type'] == 'REGULAR')]
        victims_high_value = medical_triages[(medical_triages['data.type'] == 'CRITICAL')]
        victims_low_value['Class'] = None
        victims_high_value['Class'] = None
        # victims_low_value = medical_triages[(medical_triages['data.target_block_type'].isin(['Victim Block 1', 'asistmod:block_victim_1']))]
        # victims_high_value = medical_triages[(medical_triages['data.target_block_type'].isin(['Victim Block 2', 'asistmod:block_victim_2']))]
        
        if len(victims_low_value) > 0:
            victims_low_value['Class'] = victims_low_value.apply(lambda row: victim_classification(row['data.victim_id'], victim_classes), axis=1,
                                                    result_type="expand")
            low_value_victim_class1 = len(victims_low_value[victims_low_value['Class']== 1]) 
            low_value_victim_class2 = len(victims_low_value[victims_low_value['Class']== 2])
            low_value_victim_class3 = len(victims_low_value[victims_low_value['Class']== 3])
            low_value_victim_class4 = len(victims_low_value[victims_low_value['Class']== 4]) 
            low_value_victim_class5 = len(victims_low_value[victims_low_value['Class']== 5]) 
            low_value_victim_class6 = len(victims_low_value[victims_low_value['Class']== 6]) 
            low_value_victim_class7 = len(victims_low_value[victims_low_value['Class']== 7]) 
            low_value_victim_class8 = len(victims_low_value[victims_low_value['Class']== 8]) 
            low_value_victim_class9 = len(victims_low_value[victims_low_value['Class']== 9]) 
            low_value_victim_other_classes = len(victims_low_value[victims_low_value['Class']== 0]) 
            
        else:
            low_value_victim_class1 = 0 
            low_value_victim_class2 = 0
            low_value_victim_class3 = 0
            low_value_victim_class4 = 0
            low_value_victim_class5 = 0
            low_value_victim_class6 = 0 
            low_value_victim_class7 = 0
            low_value_victim_class8 = 0 
            low_value_victim_class9 = 0
            low_value_victim_other_classes = 0
            
        
        if len(victims_high_value)>0:
            victims_high_value['Class'] = victims_high_value.apply(lambda row: victim_classification(row['data.victim_id'], victim_classes), axis=1,
                                                    result_type="expand")
            high_value_victim_class1 = len(victims_high_value[victims_high_value['Class']== 1]) 
            high_value_victim_class2 = len(victims_high_value[victims_high_value['Class']== 2]) 
            high_value_victim_class3 = len(victims_high_value[victims_high_value['Class']== 3]) 
            high_value_victim_class4 = len(victims_high_value[victims_high_value['Class']== 4]) 
            high_value_victim_other_classes = len(victims_high_value[victims_high_value['Class']== 0]) 
            
        else:
            high_value_victim_class1 = 0
            high_value_victim_class2 = 0
            high_value_victim_class3 = 0
            high_value_victim_class4 = 0
            high_value_victim_other_classes = 0

        rubble_broken = df[(df['msg.sub_type'] == 'Event:RubbleDestroyed')]
        if (len(rubble_broken) == 0) & ('HAMMER' in set(df['data.tool_type'])):
            rubble_broken = df[(df['data.target_block_type'] == 'minecraft:gravel')]

        if len(rubble_broken)>0:
            rubble_broken['Class'] = rubble_broken.apply(lambda row: rubble_classification(row['data.rubble_x'], row['data.rubble_z'], rubble_classes), axis=1,
                                                        result_type="expand")            
            rubble_class1 = len(rubble_broken[rubble_broken['Class'] == 1])
            rubble_class2 = len(rubble_broken[rubble_broken['Class'] == 2])
            rubble_class3 = len(rubble_broken[rubble_broken['Class'] == 3])
        else:
            rubble_class1 = 0
            rubble_class2 = 0
            rubble_class3 = 0

        victim_picked = df[(df['msg.sub_type'] == 'Event:VictimPickedUp')]
        # rubble_broken = rubble_broken.drop_duplicates(subset=['data.rubble_x', 'data.rubble_y', 'data.rubble_z'], keep='last')
        if len(victim_picked) > 0:
            victim_picked['Class'] = victim_picked.apply(lambda row: victim_classification(row['data.victim_id'], victim_classes), axis=1,
                                                    result_type="expand")
            low_value_victim_picked_class1 = len(victim_picked[victim_picked['Class']== 1]) 
            low_value_victim_picked_class2 = len(victim_picked[victim_picked['Class']== 2]) 
            low_value_victim_picked_class3 = len(victim_picked[victim_picked['Class']== 3]) 
            low_value_victim_picked_class4 = len(victim_picked[victim_picked['Class']== 4]) 
            low_value_victim_picked_class5 = len(victim_picked[victim_picked['Class']== 5]) 
            low_value_victim_picked_class6 = len(victim_picked[victim_picked['Class']== 6]) 
            low_value_victim_picked_class7 = len(victim_picked[victim_picked['Class']== 7]) 
            low_value_victim_picked_class8 = len(victim_picked[victim_picked['Class']== 8]) 
            low_value_victim_picked_class9 = len(victim_picked[victim_picked['Class']== 9]) 
            low_value_victim_picked_other_classes = len(victim_picked[victim_picked['Class']== 0])
            
        else:
            low_value_victim_picked_class1 = 0 
            low_value_victim_picked_class2 = 0
            low_value_victim_picked_class3 = 0
            low_value_victim_picked_class4 = 0
            low_value_victim_picked_class5 = 0
            low_value_victim_picked_class6 = 0 
            low_value_victim_picked_class7 = 0
            low_value_victim_picked_class8 = 0 
            low_value_victim_picked_class9 = 0
            low_value_victim_picked_other_classes = 0

        rubble_broken_num = len(rubble_broken)
        victim_picked_num = len(victim_picked)

        low_value_triaged_num = len(victims_low_value)
        high_value_triaged_num = len(victims_high_value)
        # points_ratio = float(yellows_triaged * 30)/(greens_triaged * 10) if greens_triaged !=0 else str(yellows_triaged * 30) + ':' + str(greens_triaged * 10)

        low_value_triaged_points = low_value_triaged_num * 10
        high_value_triaged_points = high_value_triaged_num * 50
        total_points = low_value_triaged_points + high_value_triaged_points

        df_marker = df[(df['msg.sub_type'] == 'Event:MarkerPlaced')]
        m1 = df_marker[df_marker['data.type'] == 'Marker Block 1']
        m2 = df_marker[df_marker['data.type'] == 'Marker Block 2']
        m3 = df_marker[df_marker['data.type'] == 'Marker Block 3']
        m4 = df_marker[df_marker['data.type'] == 'Marker Block 4']
        m5 = df_marker[df_marker['data.type'] == 'Marker Block 5']
        m6 = df_marker[df_marker['data.type'] == 'Marker Block 6']

        df_frozen_states = df[df['msg.sub_type'] == 'Event:PlayerFrozenStateChange']
        if len(df_frozen_states) >0:
            df_frozen = df_frozen_states[df_frozen_states['data.state_changed_to'] == 'FROZEN']
            df_unfrozen = df_frozen_states[df_frozen_states['data.state_changed_to'] == 'UNFROZEN']
            frozen_num = len(df_frozen)
            unfrozen_num = len(df_unfrozen)
        else:
            frozen_num = 0
            unfrozen_num = 0

        m_total_number = len(df_marker)
        m1_num = len(m1)
        m2_num = len(m2)
        m3_num = len(m3)
        m4_num = len(m4)
        m5_num = len(m5)
        m6_num = len(m6)

        df_tool_depleted = df[(df['msg.sub_type'] == 'Event:ToolDepleted')]
        tool_depleted_num = len(df_tool_depleted)
        if tag =='Team_':
            distance_total = 0
        else:
            df = df[(df['msg.sub_type'] == 'state')]
            if len(df) > 0:
                distance_total = compute_distance(df)
            else:
                distance_total = 0

        df_transport = df[(df['msg.sub_type'].isin(['Event:VictimPickedUp', 'Event:VictimPlaced']))]
        transport_distance = []
        for i in range(len(df_transport)//2):
            x1, z1 = df_transport.iloc[2*i][['data.victim_x','data.victim_z']]
            x2, z2 = df_transport.iloc[2*i+1][['data.victim_x', 'data.victim_z']]
            trans_distance = np.sqrt((x2-x1) ** 2 + (z2-z1) ** 2)
            transport_distance.append(trans_distance)
        total_transport_distance = np.sum(transport_distance)

    except:
        total_points = -99
        low_value_triaged_points = -99
        high_value_triaged_points = -99
        # points_ratio = -99
        # green_triaged_ratio = -99
        rubble_broken_num = -99
        victim_picked_num = -99
        low_value_triaged_num = -99
        high_value_triaged_num = -99
        m_total_number = -99
        m1_num = -99
        m2_num = -99
        m3_num = -99
        m4_num = -99
        m5_num = -99
        m6_num = -99
        frozen_num = -99
        unfrozen_num = -99
        tool_depleted_num = -99

    saved_victim_data = [[total_points, low_value_triaged_num, low_value_triaged_points, high_value_triaged_num, high_value_triaged_points, rubble_broken_num, victim_picked_num,
                          m_total_number, m1_num, m2_num, m3_num, m4_num, m5_num, m6_num, frozen_num, unfrozen_num, tool_depleted_num, distance_total,total_transport_distance,
                          low_value_victim_class1, low_value_victim_class2, low_value_victim_class3, low_value_victim_class4, low_value_victim_class5, low_value_victim_class6,
                          low_value_victim_class7, low_value_victim_class8, low_value_victim_class9, low_value_victim_other_classes,
                          high_value_victim_class1, high_value_victim_class2, high_value_victim_class3, high_value_victim_class4, high_value_victim_other_classes,
                          low_value_victim_picked_class1,low_value_victim_picked_class2,low_value_victim_picked_class3,low_value_victim_picked_class4,
                          low_value_victim_picked_class5, low_value_victim_picked_class6,low_value_victim_picked_class7,low_value_victim_picked_class8,low_value_victim_picked_class9,
                          low_value_victim_picked_other_classes, rubble_class1, rubble_class2, rubble_class3]]
    saved_victims_df = pd.DataFrame.from_records(saved_victim_data,
                                                 columns=['Total_points', 'VictimRescues_LowValue', 'VictimRescues_LowValue_points',
                                                          'VictimRescues_HighValue', 'VictimRescues_HighValue_points',
                                                          'Rubble_broken','Victims_transports',
                                                          'Markers_placed_total', 'Markers_1_placed', 'Markers_2_placed', 'Markers_3_placed', 'Markers_4_placed', 'Markers_5_placed','Markers_6_placed',
                                                          'Frozen_number', 'Unfrozen_number', 'Tool_depleted_number', 'Distance_total', 'Distance_transporting_total',
                                                          'Low_value_victim_class1', 'Low_value_victim_class2', 'Low_value_victim_class3',
                                                          'Low_value_victim_class4', 'Low_value_victim_class5', 'Low_value_victim_class6',
                                                          'Low_value_victim_class7', 'Low_value_victim_class8', 'Low_value_victim_class9', 'Low_value_victim_other_classes',
                                                          'High_value_victim_class1', 'High_value_victim_class2', 'High_value_victim_class3', 'High_value_victim_class4', 'High_value_victim_other_classes',
                                                          'Low_value_victim_class1_transport', 'Low_value_victim_class2_transport', 'Low_value_victim_class3_transport',
                                                          'Low_value_victim_class4_transport', 'Low_value_victim_class5_transport', 'Low_value_victim_class6_transport',
                                                          'Low_value_victim_class7_transport', 'Low_value_victim_class8_transport', 'Low_value_victim_class9_transport', 'Low_value_victim_other_classes_transport',
                                                          'Rubble_class1', 'Rubble_class2', 'Rubble_class3'
                                                          ])
    return saved_victims_df


def extract_triage_variables_789(df, tag):
    try:
        # df['msg.timestamp'] = pd.to_datetime(df['msg.timestamp'])
        # df['msg.timestamp'] = df['msg.timestamp'].dt.tz_localize(None)
        # df = df[df['data.mission_timer'] != "Mission Timer not initialized."]
        # df = df[df['data.mission_timer'].notna()]
        # # # half_time = df[df['msg.sub_type'] == 'Event:VictimsExpired']['msg.timestamp'].values[0]
        # # # df['segment'] = df.apply(lambda row: 1 if row['msg.timestamp'] <  half_time else 2, axis=1)
        # df = df.sort_values(by='msg.timestamp')

        successful_triages = df[(df['msg.sub_type'] == 'Event:Triage') & (df['data.triage_state'] == 'SUCCESSFUL')]
        successful_triages = successful_triages.drop_duplicates(subset=['data.victim_x', 'data.victim_z'], keep='last')
        # medical_triages = df[(df['data.tool_type'].isin(['medicalkit', 'MEDKIT']))]
        # medical_triages = medical_triages.drop_duplicates(subset=['data.target_block_x', 'data.target_block_z'],
        #                                                   keep='last')
        victims_low_value = successful_triages[
            (successful_triages['data.color'].isin(['Green', 'GREEN']))]
        victims_high_value = successful_triages[
            (successful_triages['data.color'].isin(['Yellow', 'YELLOW']))]
        rubble_broken = df[(df['msg.sub_type'] == 'Event:RubbleDestroyed')]
        if (len(rubble_broken) == 0) & ('HAMMER' in set(df['data.tool_type'])):
            rubble_broken = df[(df['data.target_block_type'] == 'minecraft:gravel')]

        victim_picked = df[(df['msg.sub_type'] == 'Event:VictimPickedUp')]
        # rubble_broken = rubble_broken.drop_duplicates(subset=['data.rubble_x', 'data.rubble_y', 'data.rubble_z'], keep='last')

        rubble_broken_num = len(rubble_broken)
        victim_picked_num = len(victim_picked)
        # num_low_value_victims = 50
        # num_high_value_victims = 5
        # yellow_triages = successful_triages[successful_triages['data.color'] == 'Yellow']
        # if len(victims_low_value) == num_low_value_victims:
        #     all_low_value_cleared = victims_low_value.iloc[[-1]]['msg.timestamp'].values[0]
        # else:
        #     all_yellow_cleared = df[df['segment'] == 2]['msg.timestamp'].min()

        # saved_victims = medical_triages['data.target_block_type'].tolist()
        # triage_order = "".join([s[0] for s in saved_victims])
        low_value_triaged_num = len(victims_low_value)
        high_value_triaged_num = len(victims_high_value)
        # points_ratio = float(yellows_triaged * 30)/(greens_triaged * 10) if greens_triaged !=0 else str(yellows_triaged * 30) + ':' + str(greens_triaged * 10)

        low_value_triaged_points = low_value_triaged_num * 10
        high_value_triaged_points = high_value_triaged_num * 50
        total_points = low_value_triaged_points + high_value_triaged_points

        # greens_triaged_before_all_yellow = len(successful_triages[(successful_triages['data.color'] == 'Green') & (successful_triages['msg.timestamp'] < all_yellow_cleared)])
        # greens_triaged_after_all_yellow = len(successful_triages[(successful_triages['data.color'] == 'Green') & (successful_triages['msg.timestamp'] >= all_yellow_cleared)])
        # green_triaged_ratio = float(greens_triaged_after_all_yellow/greens_triaged_before_all_yellow) if greens_triaged_before_all_yellow != 0 else str(greens_triaged_after_all_yellow) + ':' + str(greens_triaged_before_all_yellow)
    except:
        total_points = -99
        low_value_triaged_points = -99
        high_value_triaged_points = -99
        # points_ratio = -99
        # green_triaged_ratio = -99
        rubble_broken_num = -99
        victim_picked_num = -99
        low_value_triaged_num = -99
    saved_victim_data = [[total_points, low_value_triaged_num, low_value_triaged_points, high_value_triaged_num,
                          high_value_triaged_points, rubble_broken_num, victim_picked_num]]
    saved_victims_df = pd.DataFrame.from_records(saved_victim_data,
                                                 columns=['Total_points', 'VictimRescues_LowValue',
                                                          'VictimRescues_LowValue_points',
                                                          'VictimRescues_HighValue', 'VictimRescues_HighValue_points',
                                                          'Rubble_broken',
                                                          'Victims_transports'])
    return saved_victims_df