import pandas as pd
import datetime
import numpy as np
import os.path
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D


def plot_role(df, time_list, role_list, time_line, y, ax):
    start = time_line[0]
    end = time_line[-1]
    # tt = (end - start).total_seconds() / 900
    tt = (end - start).total_seconds() / 60
    if len(time_list)>0:
        for i in range(len(time_list)):
            # x1 = (time_list[i][0][1] - start).total_seconds()/60 * tt
            # x2 = (time_list[i][1][1] - start).total_seconds()/60 * tt
            x1 = (time_list[i][0][1] - start).total_seconds()/60
            x2 = (time_list[i][1][1] - start).total_seconds()/60
            if len(role_list) > 0:
                if role_list[i][0] in (['medical', 'Medical_Specialist']):
                    color = 'pink'
                    l = 'medical'
                elif role_list[i][0] in (['engineer', 'Hazardous_Material_Specialist']):
                    color = 'skyblue'
                    l = 'engineer'
                elif role_list[i][0] == 'None':
                    df_none = df[(df['msg.timestamp'] <= time_list[i][1][1]) & (time_list[i][0][1] <= df['msg.timestamp'])]
                    df_tool = df_none[df_none['msg.sub_type']=='Event:ToolUsed']
                    if len(df_tool) ==0:
                        color = 'grey'
                        l = 'none'
                    else:
                        tool_type = df_tool.iloc[0]['data.tool_type']
                        if tool_type in (['STRETCHER', 'STRETCHER_OCCUPIED']):
                            color = 'yellow'
                            l = 'search'
                        elif tool_type in (['HAMMER', 'hammer']):
                            color = 'skyblue'
                            l = 'engineer'
                        elif tool_type in (['medicalkit', 'MEDKIT']):
                            color = 'pink'
                            l = 'medical'
                else:
                    color = 'yellow'
                    l = 'search'
            else:
                color = 'yellow'
                l = 'search'
            ax.plot([x1, x2], [y, y], c=color, label=l,linewidth=15, zorder=0)
            # ax.plot([x1, x2], [y, y], c=color, label=l, linewidth=5, zorder=0)
            labels = ['Alpha', 'Bravo', 'Delta']
            plt.yticks([1.5, 3.5, 5.5], labels)


def plot_actions(df,time_line,y,ax):
    start = time_line[0]
    end = time_line[-1]
    # tt = (end - start).total_seconds() / 900
    tt = (end - start).total_seconds() / 60

    x_hammer = []
    x_medical = []
    x_pick = []
    x_place = []
    x_tool_depleted = []
    x_tool_replenish = []
    x_frozen = []
    x_unfrozen = []

    # df_hammer = df[(df['data.tool_type'].isin(['HAMMER', 'hammer']))]
    df_hammer = df[(df['msg.sub_type'] == 'Event:RubbleDestroyed')]
    # df_medical = df[(df['data.tool_type'].isin(['medicalkit', 'MEDKIT']))]
    df_medical = df[(df['msg.sub_type'] == 'Event:Triage') & (df['data.triage_state'] == 'SUCCESSFUL')]
    df_pick = df[(df['msg.sub_type'] == 'Event:VictimPickedUp')]
    df_place = df[(df['msg.sub_type'] == 'Event:VictimPlaced')]
    df_tool_depleted = df[(df['msg.sub_type'] == 'Event:ToolDepleted')]
    # df_tool_replenish = df[(df['msg.sub_type'] == 'Event:RoleSelected') & (df['data.prev_role'] == df['data.new_role'])]

    df_hammer_equipped = df[(df['data.tool_type'] == 'HAMMER') & (df['data.durability'] == 39)]
    df_medickit_equipped = df[(df['data.tool_type'] == 'MEDKIT') & (df['data.durability'] == 29)]
    df_stretcher_equipped = df[(df['data.tool_type'] == 'STRETCHER') & (df['data.durability'] == 19)]
    item_equipped = pd.concat([df_hammer_equipped, df_medickit_equipped, df_stretcher_equipped]).sort_values(
        by=["msg.timestamp"])
    df_tool_replenish =  item_equipped.loc[ item_equipped['data.tool_type'] == item_equipped['data.tool_type'].shift(-1)]

    df_frozen_states = df[df['msg.sub_type'] == 'Event:PlayerFrozenStateChange']
    if len(df_frozen_states) > 0:
        df_frozen = df_frozen_states[df_frozen_states['data.state_changed_to'] == 'FROZEN']
        df_unfrozen = df_frozen_states[df_frozen_states['data.state_changed_to'] == 'UNFROZEN']
        if len(df_frozen)>0:
            x_frozen = df_frozen['msg.timestamp'].tolist()
            for q in range(len(x_frozen)):
                x_frozen[q] = (x_frozen[q] - start).total_seconds()/60
        if len(df_unfrozen)>0:
            x_unfrozen = df_unfrozen['msg.timestamp'].tolist()
            for q in range(len(x_unfrozen)):
                x_unfrozen[q] = (x_unfrozen[q] - start).total_seconds()/60
        ax.scatter(x_frozen, [y]*len(x_frozen), c='aqua', marker='s', s=30,label='frozen', zorder=1)
        ax.scatter(x_unfrozen, [y] * len(x_unfrozen), c='grey', marker='s', s=30, label='unfrozen', zorder=1)

    x_hammer = df_hammer['msg.timestamp'].tolist()
    x_medical = df_medical['msg.timestamp'].tolist()
    x_pick = df_pick['msg.timestamp'].tolist()
    x_place = df_place['msg.timestamp'].tolist()

    if len(df_tool_depleted)>0:
        x_tool_depleted = df_tool_depleted['msg.timestamp'].tolist()
        for l in range(len(x_tool_depleted)):
            x_tool_depleted[l] = (x_tool_depleted[l] - start).total_seconds()/60
    if len(df_tool_replenish)>0:
        x_tool_replenish = df_tool_replenish['msg.timestamp'].tolist()
        for k in range(len(x_tool_replenish)):
            x_tool_replenish[k] = (x_tool_replenish[k] - start).total_seconds()/60

    for i in range(len(x_hammer)):
        x_hammer[i] = (x_hammer[i] - start).total_seconds()/60
    for j in range(len(x_medical)):
        x_medical[j] = (x_medical[j] - start).total_seconds()/60
    for m in range(len(x_pick)):
        x_pick[m] = (x_pick[m] - start).total_seconds()/60
    for n in range(len(x_place)):
        x_place[n] = (x_place[n] - start).total_seconds()/60

    ax.scatter(x_medical, [y]*len(x_medical), c='red', marker='X', s=30,label='medical_kit', zorder=1)
    ax.scatter(x_hammer, [y]*len(x_hammer), c='blue', marker='^', s=30,label='hammer', zorder=1)
    ax.scatter(x_pick, [y+0.05]*len(x_pick), c='orange', marker='*', s=30,label='picked', zorder=1)
    ax.scatter(x_place, [y-0.05]*len(x_place), c='darkred', marker='*', s=30,label='placed', zorder=1)
    ax.scatter(x_tool_depleted, [y]*len(x_tool_depleted), c='m', marker='d', s=30,label='tool_depleted', zorder=1)
    ax.scatter(x_tool_replenish, [y] * len(x_tool_replenish), c='lime', marker='d', s=30, label='tool_replenish', zorder=1)


def visualization_role(df, df_1, df_2, df_3, time_stamp_list_1, role_list_1, time_stamp_list_2, role_list_2, time_stamp_list_3,
                       role_list_3, team_id):
    time_stamps = time_stamp_list_1 + time_stamp_list_2 + time_stamp_list_3
    time_line = [x[1] for x in time_stamps]
    time_line.sort()
    time_line.insert(0, df['msg.timestamp'].iloc[0])
    time_line.append(df['msg.timestamp'].iloc[-1])
    step = 2
    b_1 = [time_stamp_list_1[i:i + step] for i in range(0, len(time_stamp_list_1), step)]
    b_2 = [time_stamp_list_2[i:i + step] for i in range(0, len(time_stamp_list_2), step)]
    b_3 = [time_stamp_list_3[i:i + step] for i in range(0, len(time_stamp_list_3), step)]
    plt.figure('Role Stability')
    plt.title('Role Stability Visualization of ' + team_id + ' on Mission A', y=-0.2)
    ax = plt.gca()
    ax.set_xlabel('time')
    ax.set_ylabel('players')
    # ax.set_xlim(0, 15)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6.5)
    plot_role(df_1, b_1,role_list_1,time_line,1.5,ax)
    plot_role(df_2, b_2,role_list_2,time_line,3.5,ax)
    plot_role(df_3, b_3,role_list_3,time_line,5.5,ax)
    plot_actions(df_1,time_line,1.5,ax)
    plot_actions(df_2,time_line,3.5,ax)
    plot_actions(df_3,time_line,5.5,ax)
    # plot_velocity(df_1,time_line,0,ax)
    # plot_velocity(df_2,time_line,2,ax)
    # plot_velocity(df_3,time_line,4,ax)

    plot_velocity_team(df_1,df_2,df_3,time_line,0,2,4,ax)

    # plt.legend(loc = 'upper right')
    # plt.show()
    red_patch = mpatches.Patch(color='pink', label='medical')
    blue_patch = mpatches.Patch(color='skyblue', label='engineer')
    yellow_patch = mpatches.Patch(color='yellow', label='search')
    # medical_label = Line2D([0], [0], marker='X', color='w', markerfacecolor='red', label='using medical-kit', markersize=10)
    # hammer_label = Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', label='using hammer', markersize=10)
    medical_label = Line2D([0], [0], marker='X', color='w', markerfacecolor='red', label='victim rescued', markersize=10)
    hammer_label = Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', label='rubble destroyed', markersize=10)
    pick_label = Line2D([0], [0], marker='*', color='orange', markerfacecolor='orange',linestyle='None', label='victim picked', markersize=10)
    place_label = Line2D([0], [0], marker='*', color='darkred', markerfacecolor='darkred',linestyle='None', label='victim placed', markersize=10)
    t_d_label = Line2D([0], [0], marker='d', color='w', markerfacecolor='m',linestyle='None', label='tool depleted', markersize=10)
    t_r_label = Line2D([0], [0], marker='d', color='w', markerfacecolor='lime',linestyle='None', label='tool replenished', markersize=10)
    frozen_label = Line2D([0], [0], marker='s', color='w', markerfacecolor='aqua',linestyle='None', label='frozen', markersize=10)
    unfrozen_label = Line2D([0], [0], marker='s', color='w', markerfacecolor='grey',linestyle='None', label='unfrozen', markersize=10)

    v_label = Line2D([0], [0], color='grey', lw=2, label='velocity')
    plt.legend(handles=[red_patch, medical_label, t_d_label, blue_patch, hammer_label,  t_r_label, yellow_patch, pick_label, frozen_label, v_label, place_label, unfrozen_label], bbox_to_anchor=(0.5, 1.25), loc='upper center', ncol=4)
    # plt.legend(handles=[red_patch, blue_patch, yellow_patch,  medical_label, hammer_label, pick_label, place_label, v_label],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

    save_path = 'results/figures/role_stability_' + team_id + '.jpg'
    # plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.cla()


def normalization(x,max,min):
	x = (x - min) / (max - min)
	return x


def plot_velocity(df, time_line, y, ax):
    start = time_line[0]
    end = time_line[-1]
    # tt = (end - start).total_seconds() / 900
    tt = (end - start).total_seconds() / 60
    df = df[(df['msg.sub_type'] == 'state')]
    df = df.drop_duplicates('msg.timestamp', keep='first')
    df['v'] = None
    for i in range(len(df)):
        if i == 0:
            df['v'].iloc[i] = 0
            continue
        else:
            t_i = (df['msg.timestamp'].iloc[i]-df['msg.timestamp'].iloc[i-1]).total_seconds()
            s_i = np.sqrt(np.diff([df['data.x'].iloc[i], df['data.x'].iloc[i-1]]) ** 2 + np.diff([df['data.z'].iloc[i], df['data.z'].iloc[i-1]]) ** 2)
            df['v'].iloc[i] = s_i/t_i
    x_v = df['msg.timestamp'].tolist()
    y_v = df['v'].tolist()
    for i in range(len(x_v)):
        x_v[i] = (x_v[i] - start).total_seconds()/60
    for i in range(len(y_v)):
        if i==0:
            continue
        elif y_v[i] > 100:
            y_v[i] = 0
        else:
            y_v[i] = y_v[i][0]
    y_v = [normalization(i,max(y_v),min(y_v))+y for i in y_v]
    ax.plot(x_v, y_v, c='grey',label='velocity', zorder=-1)


def plot_velocity_team(df1,df2,df3, time_line, y1,y2,y3, ax):
    start = time_line[0]
    end = time_line[-1]
    # tt = (end - start).total_seconds() / 900
    tt = (end - start).total_seconds() / 60
    df1 = df1[(df1['msg.sub_type'] == 'state')]
    df2 = df2[(df2['msg.sub_type'] == 'state')]
    df3 = df3[(df3['msg.sub_type'] == 'state')]
    df1 = df1.drop_duplicates('msg.timestamp', keep='first')
    df2 = df2.drop_duplicates('msg.timestamp', keep='first')
    df3 = df3.drop_duplicates('msg.timestamp', keep='first')
    df1['v'] = None
    df2['v'] = None
    df3['v'] = None
    for i in range(len(df1)):
        if i == 0:
            df1['v'].iloc[i] = 0
            continue
        else:
            t_i = (df1['msg.timestamp'].iloc[i]-df1['msg.timestamp'].iloc[i-1]).total_seconds()
            s_i = np.sqrt(np.diff([df1['data.x'].iloc[i], df1['data.x'].iloc[i-1]]) ** 2 + np.diff([df1['data.z'].iloc[i], df1['data.z'].iloc[i-1]]) ** 2)
            df1['v'].iloc[i] = s_i/t_i

    for j in range(len(df2)):
        if j == 0:
            df2['v'].iloc[j] = 0
            continue
        else:
            t_j = (df2['msg.timestamp'].iloc[j]-df2['msg.timestamp'].iloc[j-1]).total_seconds()
            s_j = np.sqrt(np.diff([df2['data.x'].iloc[j], df2['data.x'].iloc[j-1]]) ** 2 + np.diff([df2['data.z'].iloc[j], df2['data.z'].iloc[j-1]]) ** 2)
            df2['v'].iloc[j] = s_j/t_j

    for k in range(len(df3)):
        if k == 0:
            df3['v'].iloc[k] = 0
            continue
        else:
            t_k = (df3['msg.timestamp'].iloc[k]-df3['msg.timestamp'].iloc[k-1]).total_seconds()
            s_k = np.sqrt(np.diff([df3['data.x'].iloc[k], df3['data.x'].iloc[k-1]]) ** 2 + np.diff([df3['data.z'].iloc[k], df3['data.z'].iloc[k-1]]) ** 2)
            df3['v'].iloc[k] = s_k/t_k

    x_v_1 = df1['msg.timestamp'].tolist()
    y_v_1 = df1['v'].tolist()

    x_v_2 = df2['msg.timestamp'].tolist()
    y_v_2 = df2['v'].tolist()

    x_v_3 = df3['msg.timestamp'].tolist()
    y_v_3 = df3['v'].tolist()

    for i in range(len(x_v_1)):
        x_v_1[i] = (x_v_1[i] - start).total_seconds()/60
    for i in range(len(y_v_1)):
        if i==0:
            continue
        elif y_v_1[i] > 30:
            y_v_1[i] = 0
        else:
            y_v_1[i] = y_v_1[i][0]

    for j in range(len(x_v_2)):
        x_v_2[j] = (x_v_2[j] - start).total_seconds()/60
    for j in range(len(y_v_2)):
        if j==0:
            continue
        elif y_v_2[j] > 30:
            y_v_2[j] = 0
        else:
            y_v_2[j] = y_v_2[j][0]

    for k in range(len(x_v_3)):
        x_v_3[k] = (x_v_3[k] - start).total_seconds()/60
    for k in range(len(y_v_3)):
        if k==0:
            continue
        elif y_v_3[k] > 30:
            y_v_3[k] = 0
        else:
            y_v_3[k] = y_v_3[k][0]

    v_max = max(max(y_v_1),max(y_v_2),max(y_v_3))
    v_min = min(min(y_v_1),min(y_v_2),min(y_v_3))
    y_v_1_n = [normalization(i,v_max,v_min)+y1 for i in y_v_1]
    ax.plot(x_v_1, y_v_1_n, c='grey',label='velocity', zorder=-1)

    y_v_2_n = [normalization(i,v_max,v_min)+y2 for i in y_v_2]
    ax.plot(x_v_2, y_v_2_n, c='grey',label='velocity', zorder=-1)

    y_v_3_n = [normalization(i,v_max,v_min)+y3 for i in y_v_3]
    ax.plot(x_v_3, y_v_3_n, c='grey',label='velocity', zorder=-1)


def plot_sentiment(df, time_line, y, ax):
    start = time_line[0]
    end = time_line[-1]
    x_v = df['timestamp'].tolist()
    y_v = df['sentiment_score'].tolist()
    for i in range(len(x_v)):
        x_v[i] = (x_v[i] - start).total_seconds()/60
    # y_v = [normalization(i,max(y_v),min(y_v))+y for i in y_v]
    y_v = [i+y for i in y_v]
    ax.plot(x_v, y_v, c='grey',label='sentiment_score', zorder=-1)


def plot_dac(df,time_line,y,ax):
    start = time_line[0]
    end = time_line[-1]
    # tt = (end - start).total_seconds() / 900
    tt = (end - start).total_seconds() / 60
    
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    df = df.sort_values(by='timestamp')

    x_sd = []
    x_b = []
    x_qy = []
    x_percent = []
    x_sv = []
    x_qw = []

    df_sd = df[(df['DA'] == 'sd')]
    df_b = df[(df['DA'] == 'b')]
    df_qy = df[(df['DA'] == 'qy')]
    df_sv = df[(df['DA'] == 'sv')]
    df_qw = df[(df['DA'] == 'qw')]
    df_percent = df[(df['DA'] == '%')]

    x_sd = df_sd['timestamp'].tolist()
    x_b = df_b['timestamp'].tolist()
    x_qy = df_qy['timestamp'].tolist()
    x_sv = df_sv['timestamp'].tolist()
    x_qw = df_qw['timestamp'].tolist()
    x_percent = df_percent['timestamp'].tolist()

    for i in range(len(x_sd)):
        x_sd[i] = (x_sd[i] - start).total_seconds()/60
    for i in range(len(x_b)):
        x_b[i] = (x_b[i] - start).total_seconds()/60  
    for i in range(len(x_qy)):
        x_qy[i] = (x_qy[i] - start).total_seconds()/60
    for i in range(len(x_sv)):
        x_sv[i] = (x_sv[i] - start).total_seconds()/60
    for i in range(len(x_qw)):
        x_qw[i] = (x_qw[i] - start).total_seconds()/60
    for i in range(len(x_percent)):
        x_percent[i] = (x_percent[i] - start).total_seconds()/60
        
    ax.scatter(x_sd, [y]*len(x_sd), c='red', marker='X', s=30, zorder=1)
    ax.scatter(x_b, [y]*len(x_b), c='blue', marker='^', s=30, zorder=1)
    ax.scatter(x_qy, [y]*len(x_qy), c='orange', marker='*', s=30, zorder=1)
    ax.scatter(x_sv, [y]*len(x_sv), c='darkred', marker='o', s=30, zorder=1)
    ax.scatter(x_qw, [y]*len(x_qw), c='m', marker='d', s=30, zorder=1)
    ax.scatter(x_percent, [y] * len(x_percent), c='lime', marker='s', s=30, zorder=1)
    

def visualization_dac(df, df_1, df_2, df_3, time_stamp_list_1, role_list_1, time_stamp_list_2, role_list_2, time_stamp_list_3,
                       role_list_3, team_id, df_dac, p1, p2, p3):
    time_stamps = time_stamp_list_1 + time_stamp_list_2 + time_stamp_list_3
    time_line = [x[1] for x in time_stamps]
    time_line.sort()
    time_line.insert(0, df['msg.timestamp'].iloc[0])
    time_line.append(df['msg.timestamp'].iloc[-1])
    step = 2
    b_1 = [time_stamp_list_1[i:i + step] for i in range(0, len(time_stamp_list_1), step)]
    b_2 = [time_stamp_list_2[i:i + step] for i in range(0, len(time_stamp_list_2), step)]
    b_3 = [time_stamp_list_3[i:i + step] for i in range(0, len(time_stamp_list_3), step)]
    plt.figure('Role DAC')
    plt.title('Role DAC Visualization of ' + team_id + ' on Mission A', y=-0.2)
    ax = plt.gca()
    ax.set_xlabel('time')
    ax.set_ylabel('players')
    # ax.set_xlim(0, 15)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6.5)
    plot_role(df_1, b_1,role_list_1,time_line,1.5,ax)
    plot_role(df_2, b_2,role_list_2,time_line,3.5,ax)
    plot_role(df_3, b_3,role_list_3,time_line,5.5,ax)
    
    df_dac_1 = df_dac[(df_dac['participant_id'] == ' ' + p1)]  # player_1
    df_dac_2 = df_dac[(df_dac['participant_id'] == ' ' + p2)]  # player_2
    df_dac_3 = df_dac[(df_dac['participant_id'] == ' ' + p3)]  # player_3
    
    plot_dac(df_dac_1,time_line,1.5,ax)
    plot_dac(df_dac_2,time_line,3.5,ax)
    plot_dac(df_dac_3,time_line,5.5,ax)

    plot_sentiment(df_dac_1,time_line,0.5, ax)
    plot_sentiment(df_dac_2,time_line,2.5, ax)
    plot_sentiment(df_dac_3,time_line,4.5, ax)

    red_patch = mpatches.Patch(color='pink', label='medical')
    blue_patch = mpatches.Patch(color='skyblue', label='engineer')
    yellow_patch = mpatches.Patch(color='yellow', label='search')

    sd_label = Line2D([0], [0], marker='X', color='w', markerfacecolor='red', label='sd', markersize=10)
    b_label = Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', label='b', markersize=10)
    qy_label = Line2D([0], [0], marker='*', color='orange', markerfacecolor='orange',linestyle='None', label='qy', markersize=10)
    sv_label = Line2D([0], [0], marker='o', color='darkred', markerfacecolor='darkred',linestyle='None', label='sv', markersize=10)
    qw_label = Line2D([0], [0], marker='d', color='w', markerfacecolor='m',linestyle='None', label='qw', markersize=10)
    per_label = Line2D([0], [0], marker='s', color='w', markerfacecolor='lime',linestyle='None', label='%', markersize=10)
  
    senti_label = Line2D([0], [0], color='grey', lw=2, label='sentiment score')
    plt.legend(handles=[red_patch, sd_label, b_label, blue_patch, qy_label,  sv_label, yellow_patch, qw_label, senti_label, per_label], bbox_to_anchor=(0.5, 1.25), loc='upper center', ncol=4)
    # plt.legend(handles=[red_patch, blue_patch, yellow_patch,  medical_label, hammer_label, pick_label, place_label, v_label],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

    save_path = 'results/figures/role_dac_' + team_id + '.jpg'
    # plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.cla()
 
 