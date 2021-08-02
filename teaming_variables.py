import pandas as pd
import datetime
import numpy as np
import os.path
from pandas.core.frame import DataFrame
# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.lines import Line2D

def find_team_info(team_id):
    try:
        if team_id == 'TM000001':
            p1 = 'Aaronskiy1'
            p2 = 'clarkie765'
            p3 = 'Agent_eito'
        elif team_id == 'TM000002':
            p1 = 'FloopDeeDoop'
            p2 = 'wallacetheharp'
            p3 = 'intermonk'
        elif team_id == 'TM000003':
            p1 = 'ASIST6'
            p2 = 'ASIST5'
            p3 = 'valravn666'
        elif team_id == 'TM000004':
            p1 = 'ADAPTII1'
            p2 = 'ASIST4'
            p3 = 'DylanAlexander20'
        elif team_id == 'TM000005':
            p1 = 'ASIST2'
            p2 = 'ShriTata'
            p3 = 'WoodenHorse9773'
        elif team_id == 'TM000006':
            p1 = 'RCV2'
            p2 = 'BLUE_7_'
            p3 = 'ASIST3'
        elif team_id == 'TM000007':
            p1 = 'WoodenHorse9773'
            p2 = 'ASIST4'
            p3 = 'intermonk'
        elif team_id == 'TM000008':
            p1 = 'ShriTata'
            p2 = 'WoodenHorse9773'
            p3 = 'ASIST4'
        elif team_id == 'TM000009':
            p1 = 'intermonk'
            p2 = 'WoodenHorse9773'
            p3 = 'ASIST4'
        elif team_id == 'TM000010':
            p1 = 'ShriTata'
            p2 = 'clarkie765'
            p3 = 'ASIST4'
    except:
        print("error processing memeber id: ", team_id)
        print("No teem information found.")

    return p1, p2, p3

# def apply_tool(role_list, time_line):
#     start = time_line[0]
#     end = time_line[-1]
#     # tt = (end - start).total_seconds() / 900
#     tt = (end - start).total_seconds() / 60
#     if len(time_list)>0:
#         for i in range(len(time_list)):
#             # x1 = (time_list[i][0][1] - start).total_seconds()/60 * tt
#             # x2 = (time_list[i][1][1] - start).total_seconds()/60 * tt
#             x1 = (time_list[i][0][1] - start).total_seconds()/60
#             x2 = (time_list[i][1][1] - start).total_seconds()/60
#             if len(role_list) > 0:
#                 if role_list[i][0] in (['medical', 'Medical_Specialist']):
#                     color = 'pink'
#                     l = 'medical'
#                 elif role_list[i][0] in (['engineer', 'Hazardous_Material_Specialist']):
#                     color = 'skyblue'
#                     l = 'engineer'
#                 elif role_list[i][0] == 'None':
#                     df_none = df[(df['msg.timestamp'] <= time_list[i][1][1]) & (time_list[i][0][1] <= df['msg.timestamp'])]
#                     df_tool = df_none[df_none['msg.sub_type']=='Event:ToolUsed']
#                     if len(df_tool) ==0:
#                         color = 'grey'
#                         l = 'none'
#                     else:
#                         tool_type = df_tool.iloc[0]['data.tool_type']
#                         if tool_type in (['STRETCHER', 'STRETCHER_OCCUPIED']):
#                             color = 'yellow'
#                             l = 'search'
#                         elif tool_type in (['HAMMER', 'hammer']):
#                             color = 'skyblue'
#                             l = 'engineer'
#                         elif tool_type in (['medicalkit', 'MEDKIT']):
#                             color = 'pink'
#                             l = 'medical'
#                 else:
#                     color = 'yellow'
#                     l = 'search'
#             else:
#                 color = 'yellow'
#                 l = 'search'


def assign_roles(df_time, df):
    prev_role = df[(df['msg.timestamp'] == df_time)]['data.prev_role']
    new_role = df[(df['msg.timestamp'] == df_time)]['data.new_role']
    df_t = df[(df['msg.timestamp'] == df_time)]
    if len(prev_role) == 0:
        prev_role = 'None'
        new_role = 'None'
    # elif df_t['msg.sub_type'].item() == 'Event:ToolUsed':
    elif df_t['msg.sub_type'].to_list()[0] == 'Event:ToolUsed':
        if df_t['data.tool_type'].to_list()[0] in (['medicalkit', 'MEDKIT']):
            prev_role = 'None'
            new_role = 'Medical_Specialist'
        elif df_t['data.tool_type'].to_list()[0] in (['HAMMER', 'hammer']):
            prev_role = 'None'
            new_role = 'Hazardous_Material_Specialist'
        else:
            prev_role = 'None'
            new_role = 'Search_Specialist'
    else:
        prev_role = prev_role.values[0]
        new_role = new_role.values[0]
    return prev_role, new_role


def assign_time(df_time, df):
    x_coord = df[(df['msg.timestamp'] == df_time)]['data.x']
    z_coord = df[(df['msg.timestamp'] == df_time)]['data.z']
    if len(x_coord) == 0:
        x_coord = 'None'
        z_coord = 'None'
    else:
        x_coord = x_coord.values[0]
        z_coord = z_coord.values[0]
    return x_coord, z_coord


def fill_timestamp(df_time, tag):
    for i in range(len(df_time)):
        j = i
        while df_time[tag].iloc[i] == 'None':
            if j < (len(df_time) / 2):
                i += 1
            else:
                i -= 1
        else:
            df_time[tag].iloc[j] = df_time[tag].iloc[i]
    return df_time


def fill_roles(df_roles, tag):
    prev = 'data.prev_role_' + tag
    new = 'data.new_role_' + tag
    for i in range(len(df_roles)):
        if i == 0:
            continue
        elif (df_roles[prev].iloc[i] not in (['hammer', 'medical', 'search', 'Medical_Specialist', 'Search_Specialist',
                                              'Hazardous_Material_Specialist'])) & (df_roles[new].iloc[i] not in (
                ['hammer', 'medical', 'search', 'Medical_Specialist', 'Search_Specialist',
                 'Hazardous_Material_Specialist'])):
            df_roles[prev].iloc[i] = df_roles[new].iloc[i - 1]
            df_roles[new].iloc[i] = df_roles[new].iloc[i - 1]
        else:
            continue
    return df_roles


def compute_distance(x1, z1, x2, z2, x3, z3):
    distance_1_2 = np.sqrt(np.diff([x1, x2]) ** 2 + np.diff([z1, z2]) ** 2)
    distance_1_3 = np.sqrt(np.diff([x1, x3]) ** 2 + np.diff([z1, z3]) ** 2)
    distance_2_3 = np.sqrt(np.diff([x2, x3]) ** 2 + np.diff([z2, z3]) ** 2)
    distance_mean = np.mean([distance_1_2[0], distance_1_3[0], distance_2_3[0]])
    return distance_1_2[0], distance_1_3[0], distance_2_3[0], distance_mean


def compute_role_time(df_roles,df,df_pre):
    medical_time_list = []
    search_time_list = []
    engineer_time_list = []
    time_stamp_list = []
    role_list = []
    if len(df_pre)>0:
        df_roles.loc[-1] = df.iloc[0]
    df_roles.loc[-2] = df.iloc[-1]
    df_roles = df_roles.sort_values(by='msg.timestamp')
    start = df_roles['msg.timestamp'].min()
    start_p = df_roles['msg.timestamp'].min()
    time_stamp_list.append(['start', start])
    flag = 0
    df_change_role = df_roles[(df_roles['msg.sub_type'] == 'Event:RoleSelected')]
    # df_tool_used = df_roles[df_roles['data.tool_type'].isin(['hammer', 'medicalkit', 'HAMMER', 'MEDKIT']) | df_roles['msg.sub_type'].isin(['Event:VictimPickedUp', 'Event:VictimPlaced'])]
    df_tool_used = df_roles[df_roles['data.tool_type'].isin(['hammer', 'medicalkit', 'HAMMER', 'MEDKIT', 'STRETCHER', 'STRETCHER_OCCUPIED'])]
    temp = []
    if len(df_change_role) == 0:
        end = df_roles['msg.timestamp'].max()
        time_stamp_list.append(['end', end])
        time = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds()
        if df_tool_used['data.tool_type'].iloc[0] in (['medicalkit', 'MEDKIT']):
            medical_time_list.append(time)
            role_list.append(['medical', time])
        if df_tool_used['data.tool_type'].iloc[0] in (['HAMMER', 'hammer']):
            engineer_time_list.append(time)
            role_list.append(['engineer', time])
        # elif df_roles['msg.sub_type'].iloc[0] in (['Event:VictimPickedUp', 'Event:VictimPlaced']):
        if df_tool_used['data.tool_type'].iloc[0] in (['STRETCHER', 'STRETCHER_OCCUPIED']):
            search_time_list.append(time)
            role_list.append(['search', time])

    elif (len(df_tool_used) == 0) & (len(df_change_role) != 0):
        end = df_roles['msg.timestamp'].max()
        time_stamp_list.append(['end', end])
        time = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds()
        search_time_list.append(time)
        role_list.append(['search', time])
    else:
        for i in range(len(df_roles)):
            if (df_roles['data.new_role'].iloc[i] not in (
                    ['hammer', 'medical', 'search', 'Medical_Specialist', 'Search_Specialist',
                     'Hazardous_Material_Specialist'])) & (df_roles['msg.sub_type'].iloc[i] != 'Event:RoleSelected') & (
                    i != (len(df_roles) - 1)):
                if flag == 0:
                    start = min(df_roles['msg.timestamp'].iloc[i], start_p)
                    # start = max(df_roles['msg.timestamp'].iloc[i], start_p)
                    time_stamp_list.append(['start', start])
                    flag = 1
                continue

            elif df_roles['data.new_role'].iloc[i] in (
                    ['hammer', 'medical', 'search', 'Search_Specialist', 'Medical_Specialist',
                     'Hazardous_Material_Specialist']):
                if i == 0:
                    start = df_roles['msg.timestamp'].iloc[i]
                    time_stamp_list.append(['start', start])
                    flag = 1
                    continue
                if (i != (len(df_roles) - 1)) & (
                        (df_roles['data.new_role'].iloc[i] == df_roles['data.new_role'].iloc[i - 1]) | (
                        df_roles['data.new_role'].iloc[i] == df_roles['data.prev_role'].iloc[i])):
                    continue

            elif (i == len(df_roles) - 1) & (flag==0):
                start = min(df_roles['msg.timestamp'].iloc[i], start_p)
                time_stamp_list.append(['start', start])

            end = df_roles['msg.timestamp'].iloc[i]
            time_stamp_list.append(['end', end])
            time = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds()
            start_p = df_roles['msg.timestamp'].iloc[i]
            # time_stamp_list.append(['start', start])
            flag = 0

            if (df_roles['data.new_role'].iloc[i - 1] in (['medical', 'Medical_Specialist'])) | (
                    df_roles['data.prev_role'].iloc[i] in (['medical', 'Medical_Specialist'])) | (
                    df_roles['data.tool_type'].iloc[i] in (['medicalkit', 'MEDKIT'])):
                medical_time_list.append(time)
                role_list.append(['medical', time])
                continue
            if (df_roles['data.new_role'].iloc[i - 1] in (['search', 'Search_Specialist'])) | (
                    df_roles['data.prev_role'].iloc[i] in (['search', 'Search_Specialist'])) | (
                    df_roles['data.tool_type'].iloc[i] in (['STRETCHER', 'STRETCHER_OCCUPIED'])):
                search_time_list.append(time)
                role_list.append(['search', time])
                continue
            if (df_roles['data.new_role'].iloc[i - 1] in (['hammer', 'Hazardous_Material_Specialist'])) | (
                    df_roles['data.prev_role'].iloc[i] in (['hammer', 'Hazardous_Material_Specialist'])) | (
                    df_roles['data.tool_type'].iloc[i] in (['HAMMER', 'hammer'])):
                engineer_time_list.append(time)
                role_list.append(['engineer', time])
                continue

            elif i == (len(df_roles) - 1):
                if (df_tool_used['data.tool_type'].iloc[-1] in (['medicalkit', 'MEDKIT'])) | (
                        df_change_role['data.new_role'].iloc[-1] in (['medical', 'Medical_Specialist'])):
                    medical_time_list.append(time)
                    role_list.append(['medical', time])
                # if (df_roles['msg.sub_type'].iloc[0] in (['Event:VictimPickedUp', 'Event:VictimPlaced'])) | (df_change_role['data.new_role'].iloc[-1] in (['search', 'Search_Specialist'])):
                if (df_tool_used['data.tool_type'].iloc[-1] in (['STRETCHER', 'STRETCHER_OCCUPIED'])) | (
                            df_change_role['data.new_role'].iloc[-1] in (['search', 'Search_Specialist'])):
                    search_time_list.append(time)
                    role_list.append(['search', time])
                if (df_tool_used['data.tool_type'].iloc[-1] in (['HAMMER', 'hammer'])) | (
                        df_change_role['data.new_role'].iloc[-1] in (['hammer', 'Hazardous_Material_Specialist'])):
                    engineer_time_list.append(time)
                    role_list.append(['engineer', time])
            else:
                role_list.append(['None', time])
    [temp.append(i) for i in time_stamp_list if not i in temp]
    return medical_time_list, search_time_list, engineer_time_list, temp, role_list


def compute_role_time_2(df_roles, df):
    medical_time_list = []
    search_time_list = []
    engineer_time_list = []
    time_stamp_list = []
    role_list = []

    df_change_role = df_roles[(df_roles['msg.sub_type'] == 'Event:RoleSelected')]
    df_tool_used = df_roles[df_roles['data.tool_type'].isin(['hammer', 'medicalkit', 'HAMMER', 'MEDKIT']) | df_roles['msg.sub_type'].isin(['Event:VictimPickedUp', 'Event:VictimPlaced'])]
    temp = []
    df_t = df_change_role.copy()
    # df_t.loc[-1] = df.iloc[0]
    df_t.loc[-2] = df.iloc[-1]
    df_t = df_t.sort_values(by='msg.timestamp')
    start = df_t['msg.timestamp'].min()
    start_p = df_t['msg.timestamp'].min()
    time_stamp_list.append(['start', start])
    flag = 0

    if len(df_change_role) == 0:
        end = df_t['msg.timestamp'].max()
        time_stamp_list.append(['end', end])
        time = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds()
        if df_tool_used['data.tool_type'].iloc[0] in (['medicalkit', 'MEDKIT']):
            medical_time_list.append(time)
            role_list.append(['medical', time])
        if df_tool_used['data.tool_type'].iloc[0] in (['HAMMER', 'hammer']):
            engineer_time_list.append(time)
            role_list.append(['engineer', time])
        elif df_roles['msg.sub_type'].iloc[0] in (['Event:VictimPickedUp', 'Event:VictimPlaced']):
            role_list.append(['search', time])

    # elif (len(df_tool_used) == 0) & (len(df_change_role) != 0):
    #     end = df_roles['msg.timestamp'].max()
    #     time_stamp_list.append(['end', end])
    #     time = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds()
    #     search_time_list.append(time)
    #     role_list.append(['search', time])
    else:
        for i in range(len(df_t)):
            if (df_t['data.new_role'].iloc[i] not in (
                    ['hammer', 'medical', 'search', 'Medical_Specialist', 'Search_Specialist',
                     'Hazardous_Material_Specialist'])) & (df_roles['msg.sub_type'].iloc[i] != 'Event:RoleSelected') & (
                    i != (len(df_roles) - 1)):
                if flag == 0:
                    start = min(df_t['msg.timestamp'].iloc[i], start_p)
                    # start = max(df_roles['msg.timestamp'].iloc[i], start_p)
                    time_stamp_list.append(['start', start])
                    flag = 1
                continue

            elif df_t['data.new_role'].iloc[i] in (
                    ['hammer', 'medical', 'search', 'Search_Specialist', 'Medical_Specialist',
                     'Hazardous_Material_Specialist']):
                if i == 0:
                    start = df_roles['msg.timestamp'].iloc[i]
                    time_stamp_list.append(['start', start])
                    flag = 1
                    continue
                if (i != (len(df_roles) - 1)) & (
                        (df_roles['data.new_role'].iloc[i] == df_roles['data.new_role'].iloc[i - 1]) | (
                        df_roles['data.new_role'].iloc[i] == df_roles['data.prev_role'].iloc[i])):
                    continue

            elif (i == len(df_roles) - 1) & (flag==0):
                start = min(df_roles['msg.timestamp'].iloc[i], start_p)
                time_stamp_list.append(['start', start])

            end = df_roles['msg.timestamp'].iloc[i]
            time_stamp_list.append(['end', end])
            time = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds()
            start_p = df_roles['msg.timestamp'].iloc[i]
            # time_stamp_list.append(['start', start])
            flag = 0

            if (df_roles['data.new_role'].iloc[i - 1] in (['medical', 'Medical_Specialist'])) | (
                    df_roles['data.prev_role'].iloc[i] in (['medical', 'Medical_Specialist'])) | (
                    df_roles['data.tool_type'].iloc[i] in (['medicalkit', 'MEDKIT'])):
                medical_time_list.append(time)
                role_list.append(['medical', time])
                continue
            if (df_roles['data.new_role'].iloc[i - 1] in (['search', 'Search_Specialist'])) | (
                    df_roles['data.prev_role'].iloc[i] in (['search', 'Search_Specialist'])) | (
                    df_roles['data.tool_type'].iloc[i] == 'search'):
                search_time_list.append(time)
                role_list.append(['search', time])
                continue
            if (df_roles['data.new_role'].iloc[i - 1] in (['hammer', 'Hazardous_Material_Specialist'])) | (
                    df_roles['data.prev_role'].iloc[i] in (['hammer', 'Hazardous_Material_Specialist'])) | (
                    df_roles['data.tool_type'].iloc[i] in (['HAMMER', 'hammer'])):
                engineer_time_list.append(time)
                role_list.append(['engineer', time])
                continue

            elif i == (len(df_roles) - 1):
                if (df_tool_used['data.tool_type'].iloc[-1] in (['medicalkit', 'MEDKIT'])) | (
                        df_change_role['data.new_role'].iloc[-1] in (['medical', 'Medical_Specialist'])):
                    medical_time_list.append(time)
                    role_list.append(['medical', time])
                if (df_roles['msg.sub_type'].iloc[0] in (['Event:VictimPickedUp', 'Event:VictimPlaced'])) | (df_change_role['data.new_role'].iloc[-1] in (['search', 'Search_Specialist'])):
                    search_time_list.append(time)
                    role_list.append(['search', time])
                if (df_tool_used['data.tool_type'].iloc[-1] in (['HAMMER', 'hammer'])) | (
                        df_change_role['data.new_role'].iloc[-1] in (['hammer', 'Hazardous_Material_Specialist'])):
                    engineer_time_list.append(time)
                    role_list.append(['engineer', time])
            else:
                role_list.append(['None', time])
    [temp.append(i) for i in time_stamp_list if not i in temp]
    return medical_time_list, search_time_list, engineer_time_list, temp, role_list


def compute_role_time_team(df_roles):
    flag = 0
    time_list = []
    for i in range(len(df_roles)):
        role_list = [df_roles['data.new_role_1'].iloc[i], df_roles['data.new_role_2'].iloc[i],
                     df_roles['data.new_role_3'].iloc[i]]
        if role_list.count('None') == 2:
            continue
        elif (len(set(role_list)) < len(role_list)) & (i < (len(df_roles) - 1)):
            if flag == 0:
                start = df_roles['msg.timestamp'].iloc[i]
                flag = 1
            continue
        else:
            if flag == 0:
                continue
            else:
                end = df_roles['msg.timestamp'].iloc[i]
                flag = 0
                time = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds()
                time_list.append(time)
                start = df_roles['msg.timestamp'].iloc[i]
                continue
    redundant_time = np.sum(time_list)

    return redundant_time


def compute_role_time_team_sep(df_roles):
    flag = 0
    time_list_m_m_m = []
    time_list_m_m_e = []
    time_list_m_m_s = []
    time_list_e_e_e = []
    time_list_e_e_m = []
    time_list_e_e_s = []
    time_list_s_s_s = []
    time_list_s_s_m = []
    time_list_s_s_e = []
    # role_list_t = df_roles[df_roles.duplicated(['data.new_role_1', 'data.new_role_2', 'data.new_role_3']) == False]
    # role_list_t.loc[-1] = df_roles.iloc[-1]
    # role_list = role_list_t.loc[:, ['data.new_role_1', 'data.new_role_2', 'data.new_role_3']]
    role_list = df_roles.loc[:, ['data.new_role_1', 'data.new_role_2', 'data.new_role_3']]
    r_m = 'Medical_Specialist'
    r_e = 'Hazardous_Material_Specialist'
    r_s = 'Search_Specialist'
    for i in range(len(role_list)):
        n_m = (role_list.iloc[i] == r_m).tolist().count(True)
        n_e = (role_list.iloc[i] == r_e).tolist().count(True)
        n_s = (role_list.iloc[i] == r_s).tolist().count(True)
        if (role_list.iloc[i].tolist().count('None') == 2) | (len(set(role_list.iloc[i])) == len(role_list.iloc[i])):
            continue
        elif i < (len(role_list) - 1):
            start = df_roles['msg.timestamp'].iloc[i]
            end = df_roles['msg.timestamp'].iloc[i+1]
            time = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds()
            if n_m == 3:
                time_list_m_m_m.append(time)
            elif n_m == 2 and n_e == 1:
                time_list_m_m_e.append(time)
            elif n_m == 2 and n_s == 1:
                time_list_m_m_s.append(time)
            elif n_e == 3:
                time_list_e_e_e.append(time)
            elif n_e == 2 and n_m == 1:
                time_list_e_e_m.append(time)
            elif n_e == 2 and n_s == 1:
                time_list_e_e_s.append(time)
            elif n_s == 3:
                time_list_s_s_s.append(time)
            elif n_s == 2 and n_m == 1:
                time_list_s_s_m.append(time)
            else:
                time_list_s_s_e.append(time)
        else:
            break

    redundant_time_mmm = np.sum(time_list_m_m_m)
    redundant_time_mme = np.sum(time_list_m_m_e)
    redundant_time_mms = np.sum(time_list_m_m_s)
    redundant_time_eee = np.sum(time_list_e_e_e)
    redundant_time_eem = np.sum(time_list_e_e_m)
    redundant_time_ees = np.sum(time_list_e_e_s)
    redundant_time_sss = np.sum(time_list_s_s_s)
    redundant_time_ssm = np.sum(time_list_s_s_m)
    redundant_time_sse = np.sum(time_list_s_s_e)

    redundant_time_mm = redundant_time_mmm + redundant_time_mme + redundant_time_mms
    redundant_time_ee = redundant_time_eee + redundant_time_ees + redundant_time_eem
    redundant_time_ss = redundant_time_sss + redundant_time_sse + redundant_time_ssm

    return redundant_time_mm, redundant_time_mmm,redundant_time_mme,redundant_time_mms, redundant_time_ee, redundant_time_eee, redundant_time_eem, redundant_time_ees,\
           redundant_time_ss, redundant_time_sss,redundant_time_ssm,redundant_time_sse


def compute_role_time_player(df_roles, tag):
    flag = 0
    time_list_medicine = []
    time_list_engineer = []
    time_list_search = []
    new = 'data.new_role_' + tag
    prev = 'data.prev_role_' + tag
    role_list = df_roles.loc[:, ['data.new_role_1', 'data.new_role_2', 'data.new_role_3']]

    for i in range(len(df_roles)):
        dup = role_list.iloc[i].duplicated(keep=False)
        if (role_list.iloc[i].tolist().count('None') == 2) | (len(set(role_list.iloc[i])) == len(role_list.iloc[i])) | (dup[int(tag)-1] == False):
            continue
        elif i < (len(role_list) - 1):
            start = df_roles['msg.timestamp'].iloc[i]
            end = df_roles['msg.timestamp'].iloc[i+1]
            time = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds()
            if role_list.iloc[i][int(tag) - 1] == 'Medical_Specialist':
                time_list_medicine.append(time)
            elif role_list.iloc[i][int(tag) - 1] == 'Hazardous_Material_Specialist':
                time_list_engineer.append(time)
            elif role_list.iloc[i][int(tag) - 1] == 'Search_Specialist':
                time_list_search.append(time)
        else:
            break

    redundant_time_med = np.sum(time_list_medicine)
    redundant_time_eng = np.sum(time_list_engineer)
    redundant_time_search = np.sum(time_list_search)

    return redundant_time_med, redundant_time_eng, redundant_time_search

#
# def compute_role_time_player_0(df_roles, tag):
#     flag = 0
#     time_list_medicine = []
#     time_list_engineer = []
#     time_list_search = []
#     new = 'data.new_role_' + tag
#     prev = 'data.prev_role_' + tag
#     role_list = df_roles.loc[:, ['data.new_role_1', 'data.new_role_2', 'data.new_role_3']]
#
#     for i in range(len(df_roles)):
#         dup = role_list.iloc[i].duplicated(keep=False)
#         if role_list.iloc[i].tolist().count('None') == 2:
#             continue
#         elif (dup[int(tag)-1] == True) & (i < (len(df_roles) - 1)):
#             if flag == 0:
#                 start = df_roles['msg.timestamp'].iloc[i]
#                 flag = 1
#             continue
#         else:
#             if flag == 0:
#                 continue
#             else:
#                 end = df_roles['msg.timestamp'].iloc[i]
#                 flag = 0
#                 time = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds()
#                 if role_list.iloc[i-1][int(tag)-1] == 'Medical_Specialist':
#                     time_list_medicine.append(time)
#                 elif role_list.iloc[i-1][int(tag)-1] == 'Hazardous_Material_Specialist':
#                     time_list_engineer.append(time)
#                 elif role_list.iloc[i-1][int(tag)-1] == 'Search_Specialist':
#                     time_list_search.append(time)
#                 start = df_roles['msg.timestamp'].iloc[i]
#                 continue
#     redundant_time_med = np.sum(time_list_medicine)
#     redundant_time_eng = np.sum(time_list_engineer)
#     redundant_time_search = np.sum(time_list_search)
#
#     return redundant_time_med, redundant_time_eng, redundant_time_search


def compute_time(df_time, tag):
    proximity_list = []
    period_list = []
    start = df_time['msg.timestamp'].min()
    flag = 0
    if tag == 'Player1_':
        for i in range(len(df_time)):
            if i == len(df_time) - 1:
                end = df_time['msg.timestamp'].max()
            if ((df_time['distance1_2'].iloc[i] <= 20) | (df_time['distance1_3'].iloc[i] <= 20)):
                if flag == 0:
                    start = df_time['msg.timestamp'].iloc[i]
                    flag = 1
                continue
            else:
                if flag == 1:
                    end = df_time['msg.timestamp'].iloc[i]
                    period_list.append([start, end])
                else:
                    continue
            flag = 0
            time = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds()
            proximity_list.append(time)
    if tag == 'Player2_':
        for i in range(len(df_time)):
            if i == len(df_time) - 1:
                end = df_time['msg.timestamp'].max()
            if ((df_time['distance1_2'].iloc[i] <= 20) | (df_time['distance2_3'].iloc[i] <= 20)):
                if flag == 0:
                    start = df_time['msg.timestamp'].iloc[i]
                    flag = 1
                continue
            else:
                if flag == 1:
                    end = df_time['msg.timestamp'].iloc[i]
                    period_list.append([start, end])
                else:
                    continue
            flag = 0
            time = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds()
            proximity_list.append(time)

    if tag == 'Player3_':
        for i in range(len(df_time)):
            if i == len(df_time) - 1:
                end = df_time['msg.timestamp'].max()
            if ((df_time['distance1_3'].iloc[i] <= 20) | (df_time['distance2_3'].iloc[i] <= 20)):
                if flag == 0:
                    start = df_time['msg.timestamp'].iloc[i]
                    flag = 1
                continue
            else:
                if flag == 1:
                    end = df_time['msg.timestamp'].iloc[i]
                    period_list.append([start, end])
                else:
                    continue
            flag = 0
            time = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds()
            proximity_list.append(time)

    if tag == 'Team_':
        for i in range(len(df_time)):
            if i == len(df_time) - 1:
                end = df_time['msg.timestamp'].max()
            if df_time['distance_mean'].iloc[i] <= 20:
                if flag == 0:
                    start = df_time['msg.timestamp'].iloc[i]
                    flag = 1
                continue
            else:
                if flag == 1:
                    end = df_time['msg.timestamp'].iloc[i]
                    period_list.append([start, end])
                else:
                    continue
            flag = 0
            time = (pd.to_datetime(end) - pd.to_datetime(start)).total_seconds()
            proximity_list.append(time)

    proximity_sum = np.sum(proximity_list)
    return proximity_sum, period_list


def assign_proximity_role(df_time, time_stamp_list, role_list):
    role = 'None'
    for i in range(len(role_list)):
        start = time_stamp_list[i*2][1]
        end = time_stamp_list[i*2+1][1]
        if start < pd.to_datetime(df_time) < end:
            role = role_list[i][0]
        else:
            continue
    return role


def extract_proximity_variables(df, team_id, p1, p2, p3, time_stamp_list_1, role_list_1, time_stamp_list_2, role_list_2, time_stamp_list_3, role_list_3, tag):
    try:
        proximity_file_path = 'results/proximity_temp_missionB/df_time_' + team_id + '.csv'
        if os.path.isfile(proximity_file_path):
            df_prox = pd.read_csv(proximity_file_path)
        else:
            df = df[(df['msg.sub_type'] == 'state')]
            df_prox = df['msg.timestamp']
            df_prox = df_prox.drop_duplicates(keep='first')
            df_prox = df_prox.to_frame()
            # df_1 = df[(df['data.participant_id'] == p1)]  # player_1
            # df_2 = df[(df['data.participant_id'] == p2)]  # player_2
            # df_3 = df[(df['data.participant_id'] == p3)]  # player_3
            df_1 = df[(df['data.playername'] == p1)]  # player_1
            df_2 = df[(df['data.playername'] == p2)]  # player_2
            df_3 = df[(df['data.playername'] == p3)]  # player_3

            df_prox[['x_1', 'z_1']] = df_prox.apply(lambda row: assign_time(row['msg.timestamp'], df_1), axis=1,
                                                    result_type="expand")
            df_prox[['x_2', 'z_2']] = df_prox.apply(lambda row: assign_time(row['msg.timestamp'], df_2), axis=1,
                                                    result_type="expand")
            df_prox[['x_3', 'z_3']] = df_prox.apply(lambda row: assign_time(row['msg.timestamp'], df_3), axis=1,
                                                    result_type="expand")

            df_prox = fill_timestamp(df_prox, 'x_1')
            df_prox = fill_timestamp(df_prox, 'z_1')
            df_prox = fill_timestamp(df_prox, 'x_2')
            df_prox = fill_timestamp(df_prox, 'z_2')
            df_prox = fill_timestamp(df_prox, 'x_3')
            df_prox = fill_timestamp(df_prox, 'z_3')

            df_prox[['distance1_2', 'distance1_3', 'distance2_3', 'distance_mean']] = df_prox.apply(
                lambda row: compute_distance(row['x_1'], row['z_1'], row['x_2'], row['z_2'], row['x_3'], row['z_3']),
                axis=1, result_type="expand")

            df_prox.to_csv(proximity_file_path)

        # df_prox[['role_1']] = df_prox.apply(lambda row: assign_proximity_role(row['msg.timestamp'], time_stamp_list_1, role_list_1), axis=1,
        #                                             result_type="expand")
        # df_prox[['role_2']] = df_prox.apply(lambda row: assign_proximity_role(row['msg.timestamp'], time_stamp_list_2, role_list_2), axis=1,
        #                                             result_type="expand")
        # df_prox[['role_3']] = df_prox.apply(lambda row: assign_proximity_role(row['msg.timestamp'], time_stamp_list_3, role_list_3), axis=1,
        #                                             result_type="expand")

        proximity_data, period_list = compute_time(df_prox, tag)
    except:
        proximity_data = -99

    proximity_df = pd.DataFrame.from_records([[proximity_data]],
                                             columns=['Time_teammate_in_proximity'])

    return proximity_df


def extract_team_role_variables(df, p1, p2, p3, tag):
    try:
        df_tool = df[(df['msg.sub_type'] == 'Event:RoleSelected') | (df['msg.sub_type'] == 'Event:ToolUsed')]
        # df_tool = df[(df['msg.sub_type'].isin(['Event:ItemEquipped', 'Event:ToolUsed', 'Event:RoleSelected', 'Event:VictimPickedUp', 'Event:VictimPlaced']))]

        df_time = df_tool['msg.timestamp']
        df_time = df_time.to_frame()

        df_time.loc[-2] = df.iloc[-1]['msg.timestamp']
        df_time = df_time.sort_values(by='msg.timestamp')

        df_time = df_time.drop_duplicates(keep='first')

        # df_1 = df_tool[(df_tool['data.participant_id'] == p1)]  # player_1
        # df_2 = df_tool[(df_tool['data.participant_id'] == p2)]  # player_2
        # df_3 = df_tool[(df_tool['data.participant_id'] == p3)]  # player_3

        df_1 = df_tool[(df_tool['data.playername'] == p1)]  # player_1
        df_2 = df_tool[(df_tool['data.playername'] == p2)]  # player_2
        df_3 = df_tool[(df_tool['data.playername'] == p3)]  # player_3

        df_time[['data.prev_role_1', 'data.new_role_1']] = df_time.apply(
            lambda row: assign_roles(row['msg.timestamp'], df_1), axis=1,
            result_type="expand")
        df_time[['data.prev_role_2', 'data.new_role_2']] = df_time.apply(
            lambda row: assign_roles(row['msg.timestamp'], df_2), axis=1,
            result_type="expand")
        df_time[['data.prev_role_3', 'data.new_role_3']] = df_time.apply(
            lambda row: assign_roles(row['msg.timestamp'], df_3), axis=1,
            result_type="expand")

        df_time = fill_roles(df_time, '1')
        df_time = fill_roles(df_time, '2')
        df_time = fill_roles(df_time, '3')
        time_redundant_roles = compute_role_time_team(df_time)

        df1_redundant_time_medic, df1_redundant_time_eng, df1_redundant_time_search = compute_role_time_player(df_time, '1')
        df2_redundant_time_medic, df2_redundant_time_eng, df2_redundant_time_search = compute_role_time_player(df_time, '2')
        df3_redundant_time_medic, df3_redundant_time_eng, df3_redundant_time_search = compute_role_time_player(df_time, '3')

        redundant_time_mm, redundant_time_mmm, redundant_time_mme, redundant_time_mms, redundant_time_ee, redundant_time_eee, redundant_time_eem, redundant_time_ees, \
        redundant_time_ss, redundant_time_sss, redundant_time_ssm, redundant_time_sse = compute_role_time_team_sep(df_time)



    except:
        time_redundant_roles = -99
        redundant_time_mm = -99
        redundant_time_ee = -99
        redundant_time_ss = -99

    team_role_df = pd.DataFrame.from_records([[time_redundant_roles, redundant_time_mm, redundant_time_mmm, redundant_time_mme, redundant_time_mms, redundant_time_ee, redundant_time_eee, redundant_time_eem, redundant_time_ees, \
        redundant_time_ss, redundant_time_sss, redundant_time_ssm, redundant_time_sse]],
                                             columns=['Time_team_has_redundant_roles', 'Time_team_has_redundant_medic', 'Time_team_has_m_m_m', 'Time_team_has_m_m_e', 'Time_team_has_m_m_s',
                                                      'Time_team_has_redundant_engineer', 'Time_team_has_e_e_e', 'Time_team_has_e_e_m', 'Time_team_has_e_e_s',
                                                      'Time_team_has_redundant_search', 'Time_team_has_s_s_s','Time_team_has_s_s_m', 'Time_team_has_s_s_e'
                                                      ])
    role_redundant_df1 = pd.DataFrame.from_records([[df1_redundant_time_medic, df1_redundant_time_eng, df1_redundant_time_search]],
                                             columns=['Time_as_redundant_medic',
                                                      'Time_as_redundant_engineer',
                                                      'Time_as_redundant_search'
                                                      ])
    role_redundant_df2 = pd.DataFrame.from_records([[df2_redundant_time_medic, df2_redundant_time_eng, df2_redundant_time_search]],
                                             columns=['Time_as_redundant_medic',
                                                      'Time_as_redundant_engineer',
                                                      'Time_as_redundant_search'
                                                      ])
    role_redundant_df3 = pd.DataFrame.from_records([[df3_redundant_time_medic, df3_redundant_time_eng, df3_redundant_time_search]],
                                             columns=['Time_as_redundant_medic',
                                                      'Time_as_redundant_engineer',
                                                      'Time_as_redundant_search'
                                                      ])
    return team_role_df, role_redundant_df1, role_redundant_df2, role_redundant_df3


def extract_role_variables(df, df_pre, tag):
    try:
        # df_tool = df[(df['msg.sub_type'] == 'Event:RoleSelected') | (df['msg.sub_type'] == 'Event:ToolUsed')]
        df_tool = df[(df['msg.sub_type'].isin(['Event:ItemEquipped', 'Event:ToolUsed', 'Event:RoleSelected', 'Event:VictimPickedUp', 'Event:VictimPlaced']))]
        # df_tool_2 = df[(df['msg.sub_type']== 'Event:ItemEquipped')&(df['data.equippeditemname'].isin(
        #     ['asistmod:item_stretcher','asistmod:item_medical_kit','asistmod:item_hammer']))]
        df_tool_replenish = 0
        df_tool_swap = 0
        # for i in range(len(df_tool_2)):
        #     if i < 1:
        #         continue
        #     else:
        #         if df_tool_2.iloc[i]['data.equippeditemname'] == df_tool_2.iloc[i-1]['data.equippeditemname']:
        #             df_tool_replenish += 1
        #         else:
        #             df_tool_swap += 1

        df_hammer_equipped = df[(df['data.tool_type'] == 'HAMMER')&(df['data.durability']==39)]
        df_medickit_equipped = df[(df['data.tool_type'] == 'MEDKIT')&(df['data.durability']==29)]
        df_stretcher_equipped = df[(df['data.tool_type'] =='STRETCHER')&(df['data.durability']==19)]
        item_equipped = pd.concat([df_hammer_equipped , df_medickit_equipped,  df_stretcher_equipped]).sort_values(by=["msg.timestamp"])
        for i in range(len(item_equipped)):
            if i == 0:
                continue
            if item_equipped.iloc[i]['data.tool_type'] == item_equipped.iloc[i-1]['data.tool_type']:
                df_tool_replenish += 1
            else:
                df_tool_swap += 1

        # df_tool_replenish = len(
        #     df_tool[(df_tool['msg.sub_type'] == 'Event:RoleSelected') & (
        #             df_tool['data.prev_role'] == df_tool['data.new_role'])])
        # df_tool_swap = len(
        #     df_tool[(df_tool['msg.sub_type'] == 'Event:RoleSelected') & (
        #             df_tool['data.prev_role'] != df_tool['data.new_role'])])

        time_stamp_list = []
        role_list = []

        if len(df_tool) > 0:
            medical_time, search_time, engineer_time, time_stamp_list, role_list = compute_role_time(df_tool,df,df_pre)
            # medical_time, search_time, engineer_time, time_stamp_list, role_list = compute_role_time_2(df_tool, df)
            medical_time_longest = max(medical_time, default=0)
            search_time_longest = max(search_time, default=0)
            engineer_time_longest = max(engineer_time, default=0)
            medical_time_total = np.sum(medical_time)
            search_time_total = np.sum(search_time)
            engineer_time_total = np.sum(engineer_time)

            any_role_longest = max([medical_time_longest, engineer_time_longest,
                                    search_time_longest], default=0)
        else:
            df_tool_replenish = 0
            df_tool_swap = 0
            medical_time_longest = 0
            engineer_time_longest = 0
            search_time_longest = 0
            any_role_longest = 0
            medical_time_total = 0
            search_time_total = 0
            engineer_time_total = 0

    except:
        df_tool_replenish = -99
        df_tool_swap = -99
        medical_time_longest = -99
        engineer_time_longest = -99
        search_time_longest = -99
        any_role_longest = -99
        medical_time_total = -99
        search_time_total = -99
        engineer_time_total = -99

    role_data = [
        [df_tool_replenish, df_tool_swap, medical_time_longest, engineer_time_longest, search_time_longest,
         any_role_longest, medical_time_total, engineer_time_total, search_time_total]]
    role_df = pd.DataFrame.from_records(role_data,
                                        columns=['Count_replenish_current_tool', 'Count_swap_current_tool',
                                                 'Time_longest_as_medical_specialist',
                                                 'Time_longest_as_engineer',
                                                 'Time_longest_as_search_specialist',
                                                 'Time_longest_continuous_any_role',
                                                 'Time_as_MedicalSpecialist',
                                                 'Time_as_EngineerSpecialist',
                                                 'Time_as_SearchSpecialist'
                                                 ])
    return role_df, time_stamp_list, role_list