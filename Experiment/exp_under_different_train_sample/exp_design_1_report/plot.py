import pandas as pd
from matplotlib import pyplot as plt, ticker
from matplotlib.ticker import FuncFormatter, ScalarFormatter
import numpy as np


def plot_first_strand(task = 3):

    # Unbind_all = pd.read_csv(f"./Unbind_all_current_mean_var_strand.csv")
    # Know_unbind = pd.read_csv(f"./Know_unbind_current_mean_var_strandV2.csv")
    # Ablation_S1 = pd.read_csv(f"C:/Users/icelab01/Desktop/ZhuohuaLiu_2024_BYA_jiebang/BYA_jiebang/BYA_jiebang"
    #                           f"/Knowledge_unbinding/Experiment/exp_time_cost/Exp_ablation/FigureGen/Ablation_S1_current_mean_var_strand.csv")
    # Ablation_S2 = pd.read_csv(f"C:/Users/icelab01/Desktop/ZhuohuaLiu_2024_BYA_jiebang/BYA_jiebang/BYA_jiebang"
    #                           f"/Knowledge_unbinding/Experiment/exp_time_cost/Exp_ablation/FigureGen/Ablation_S2_current_mean_var_strand.csv")
    # Ablation_S3 = pd.read_csv(f"C:/Users/icelab01/Desktop/ZhuohuaLiu_2024_BYA_jiebang/BYA_jiebang/BYA_jiebang"
    #                           f"/Knowledge_unbinding/Experiment/exp_time_cost/Exp_ablation/FigureGen/Ablation_S3_current_mean_var_strand.csv")
    Proposed_OTA5 = pd.read_csv(f"./Proposed_OTA5_current_mean_var_strand.csv")
    Unbind_all_200 = pd.read_csv(f"./Unbind_all_V2_current_mean_var_strand.csv")
    RS_all_OTA_two = pd.read_csv(f"./RS_all_OTA_two_current_mean_var_strand.csv")
    SMAC_all_OTA_three = pd.read_csv(f"./SMAC_all_OTA_three_current_mean_var_strand.csv")
    Unbind_all_OTA_two = pd.read_csv(f"./Unbind_all_OTA_two_current_mean_var_strand.csv")
    Proposed_OTA_two = pd.read_csv(f"./Proposed_OTA_two_current_mean_var_strand.csv")
    Unbind_all_bandgap = pd.read_csv(f"./Unbind_all_bandgap_current_mean_var_strand.csv")
    Proposed_bandgap = pd.read_csv(f"./Proposed_bandgap_current_mean_var_strand.csv")
    Unbind_all_OTA_three = pd.read_csv(f"./Unbind_all_OTA_three_current_mean_var_strand.csv")
    SMAC_all_OTA_two = pd.read_csv(f"./SMAC_all_OTA_two_current_mean_var_strand.csv")
    SMAC_all_OTA_three = pd.read_csv(f"./SMAC_all_OTA_three_current_mean_var_strand.csv")
    SMAC_all_bandgap = pd.read_csv(f"./SMAC_all_bandgap_current_mean_var_strand.csv")
    RS_all_OTA_three = pd.read_csv(f"./RS_all_OTA_three_current_mean_var_strand.csv")
    Proposed_OTA_three = pd.read_csv(f"./Proposed_OTA_three_current_mean_var_strand.csv")
    MACE_all_OTA_two = pd.read_csv(f"./MACE_all_OTA_two_current_mean_var_strand.csv")
    Proposed_MACE_OTA_two = pd.read_csv(f"./Proposed_MACE_OTA_two_current_mean_var_strand.csv")
    MACE_all_OTA_three = pd.read_csv(f"./MACE_all_OTA_three_current_mean_var_strand.csv")
    Proposed_MACE_OTA_three = pd.read_csv(f"./Proposed_MACE_OTA_three_current_mean_var_strand.csv")
    MACE_all_bandgap = pd.read_csv(f"./MACE_all_bandgap_current_mean_var_strand.csv")
    Proposed_MACE_bandgap = pd.read_csv(f"./Proposed_MACE_bandgap_current_mean_var_strand.csv")
    REMBO_all_OTA_three = pd.read_csv(f"./REMBO_all_OTA_three_current_mean_var_strand.csv")
    Proposed_REMBO_OTA_three = pd.read_csv(f"./Proposed_REMBO_OTA_three_current_mean_var_strand.csv")
    REMBO_all_OTA_two = pd.read_csv(f"./REMBO_all_OTA_two_current_mean_var_strand.csv")
    Proposed_REMBO_OTA_two = pd.read_csv(f"./Proposed_REMBO_OTA_two_current_mean_var_strand.csv")
    REMBO_all_bandgap = pd.read_csv(f"./REMBO_all_bandgap_current_mean_var_strand.csv")
    Proposed_REMBO_bandgap = pd.read_csv(f"./Proposed_REMBO_bandgap_current_mean_var_strand.csv")
    Proposed_SMAC_OTA_two = pd.read_csv(f"./Proposed_SMAC_OTA_two_current_mean_var_strand.csv")
    Proposed_SMAC_OTA_three = pd.read_csv(f"./Proposed_SMAC_OTA_three_current_mean_var_strand.csv")
    Proposed_SMAC_bandgap = pd.read_csv(f"./Proposed_SMAC_bandgap_current_mean_var_strand.csv")
    USeMOC_all_OTA_two = pd.read_csv(f"./USeMOC_all_OTA_two_current_mean_var_strand.csv")
    Proposed_USeMOC_OTA_two = pd.read_csv(f"./Proposed_USeMOC_OTA_two_current_mean_var_strand.csv")
    USeMOC_all_OTA_three = pd.read_csv(f"./USeMOC_all_OTA_three_current_mean_var_strand.csv")
    Proposed_USeMOC_OTA_three = pd.read_csv(f"./Proposed_USeMOC_OTA_three_current_mean_var_strand.csv")
    USeMOC_all_bandgap = pd.read_csv(f"./USeMOC_all_bandgap_current_mean_var_strand.csv")
    Proposed_USeMOC_bandgap = pd.read_csv(f"./Proposed_USeMOC_bandgap_current_mean_var_strand.csv")


    label = ['Know_unbind', 'Unbind_all', 'Ablation_S1', 'Ablation_S2', 'Ablation_S3', 'Proposed_OTA5',
             'Unbind_all_200', 'RS_all_OTA_three', 'SMAC_all_OTA_three', 'Proposed_OTA_three', 'Proposed_OTA_three_matrix', 'Know_unbind_OTA_three',
             'Unbind_all_OTA_three', 'Proposed_OTA_three_FoM']
    color = ['#d62728',     # 深红色       proposed用
              '#ff7f0e',     # 橙色       传统BO用
             '#2ca02c',     # 绿色         MACE
             '#1f77b4',     # 中蓝色        REMBO
             '#9467bd',     # 深紫色       USeMOC
             '#8c564b',     # 棕色         RS
             '#e377c2',     # 淡紫红色      SMAC
             '#000000',     # 黑色
             '#17becf',     # 浅蓝色
             '#808080']     # 深灰色       
    y_label = ['ppm', 'FOM', '$\mathrm{I_{total}}$', '$\mathrm{Log(I_{total})}$', '$\mathrm{I_{total}(mA)}$']
    # stand = [4.36451864602757, -7.7983704250049115, 4.477842344390243, 17.028220203024727]

    # Unbind_all_iter_times = Unbind_all["iter_times"]
    # Unbind_all_mean = Unbind_all["mean"]
    # Unbind_all_var = Unbind_all["var"]
    # Know_unbind_iter_times = Know_unbind["iter_times"]
    # Know_unbind_mean = Know_unbind["mean"]
    # Know_unbind_var = Know_unbind["var"]
    # Ablation_S1_iter_times = Ablation_S1["iter_times"]
    # Ablation_S1_mean = Ablation_S1["mean"]
    # Ablation_S1_var = Ablation_S1["var"]
    # Ablation_S2_iter_times = Ablation_S2["iter_times"]
    # Ablation_S2_mean = Ablation_S2["mean"]
    # Ablation_S2_var = Ablation_S2["var"]
    # Ablation_S3_iter_times = Ablation_S3["iter_times"]
    # Ablation_S3_mean = Ablation_S3["mean"]
    # Ablation_S3_var = Ablation_S3["var"]
    Proposed_OTA5_iter_times = Proposed_OTA5["iter_times"]
    Proposed_OTA5_mean = Proposed_OTA5["mean"]
    Proposed_OTA5_var = Proposed_OTA5["var"]
    Unbind_all_200_iter_times = Unbind_all_200["iter_times"]
    Unbind_all_200_mean = Unbind_all_200["mean"]
    Unbind_all_200_var = Unbind_all_200["var"]
    RS_all_OTA_two_iter_times = RS_all_OTA_two["iter_times"]
    RS_all_OTA_two_mean = RS_all_OTA_two["mean"]
    RS_all_OTA_two_var = RS_all_OTA_two["var"]
    SMAC_all_OTA_two_iter_times = SMAC_all_OTA_two["iter_times"]
    SMAC_all_OTA_two_mean = SMAC_all_OTA_two["mean"]
    SMAC_all_OTA_two_var = SMAC_all_OTA_two["var"]
    Proposed_OTA_two_iter_times = Proposed_OTA_two["iter_times"]
    Proposed_OTA_two_mean = Proposed_OTA_two["mean"]
    Proposed_OTA_two_var = Proposed_OTA_two["var"]
    Unbind_all_OTA_two_iter_times = Unbind_all_OTA_two["iter_times"]
    Unbind_all_OTA_two_mean = Unbind_all_OTA_two["mean"]
    Unbind_all_OTA_two_var = Unbind_all_OTA_two["var"]
    Proposed_bandgap_iter_times = Proposed_bandgap["iter_times"]
    Proposed_bandgap_mean = Proposed_bandgap["mean"]
    Proposed_bandgap_var = Proposed_bandgap["var"]
    Unbind_all_bandgap_iter_times = Unbind_all_bandgap["iter_times"]
    Unbind_all_bandgap_mean = Unbind_all_bandgap["mean"]
    Unbind_all_bandgap_var = Unbind_all_bandgap["var"]
    # Know_unbind_OTA_three_iter_times = Know_unbind_OTA_three["iter_times"]
    # Know_unbind_OTA_three_mean = Know_unbind_OTA_three["mean"]
    # Know_unbind_OTA_three_var = Know_unbind_OTA_three["var"]
    Unbind_all_OTA_three_iter_times = Unbind_all_OTA_three["iter_times"]
    Unbind_all_OTA_three_mean = Unbind_all_OTA_three["mean"]
    Unbind_all_OTA_three_var = Unbind_all_OTA_three["var"]
    SMAC_all_OTA_three_iter_times = SMAC_all_OTA_three["iter_times"]
    SMAC_all_OTA_three_mean = SMAC_all_OTA_three["mean"]
    SMAC_all_OTA_three_var = SMAC_all_OTA_three["var"]
    SMAC_all_bandgap_iter_times = SMAC_all_bandgap["iter_times"]
    SMAC_all_bandgap_mean = SMAC_all_bandgap["mean"]
    SMAC_all_bandgap_var = SMAC_all_bandgap["var"]
    RS_all_OTA_three_iter_times = RS_all_OTA_three["iter_times"]
    RS_all_OTA_three_mean = RS_all_OTA_three["mean"]
    RS_all_OTA_three_var = RS_all_OTA_three["var"]
    Proposed_OTA_three_iter_times = Proposed_OTA_three["iter_times"]
    Proposed_OTA_three_mean = Proposed_OTA_three["mean"]
    Proposed_OTA_three_var = Proposed_OTA_three["var"]
    MACE_all_OTA_two_iter_times = MACE_all_OTA_two["iter_times"]
    MACE_all_OTA_two_mean = MACE_all_OTA_two["mean"]
    MACE_all_OTA_two_var = MACE_all_OTA_two["var"]
    Proposed_MACE_OTA_two_iter_times = Proposed_MACE_OTA_two["iter_times"]
    Proposed_MACE_OTA_two_mean = Proposed_MACE_OTA_two["mean"]
    Proposed_MACE_OTA_two_var = Proposed_MACE_OTA_two["var"]
    MACE_all_OTA_three_iter_times = MACE_all_OTA_three["iter_times"]
    MACE_all_OTA_three_mean = MACE_all_OTA_three["mean"]
    MACE_all_OTA_three_var = MACE_all_OTA_three["var"]
    Proposed_MACE_OTA_three_iter_times = Proposed_MACE_OTA_three["iter_times"]
    Proposed_MACE_OTA_three_mean = Proposed_MACE_OTA_three["mean"]
    Proposed_MACE_OTA_three_var = Proposed_MACE_OTA_three["var"]
    MACE_all_bandgap_iter_times = MACE_all_bandgap["iter_times"]
    MACE_all_bandgap_mean = MACE_all_bandgap["mean"]
    MACE_all_bandgap_var = MACE_all_bandgap["var"]
    Proposed_MACE_bandgap_iter_times = Proposed_MACE_bandgap["iter_times"]
    Proposed_MACE_bandgap_mean = Proposed_MACE_bandgap["mean"]
    Proposed_MACE_bandgap_var = Proposed_MACE_bandgap["var"]
    REMBO_all_OTA_two_iter_times = REMBO_all_OTA_two["iter_times"]
    REMBO_all_OTA_two_mean = REMBO_all_OTA_two["mean"]
    REMBO_all_OTA_two_var = REMBO_all_OTA_two["var"]
    Proposed_REMBO_OTA_two_iter_times = Proposed_REMBO_OTA_two["iter_times"]
    Proposed_REMBO_OTA_two_mean = Proposed_REMBO_OTA_two["mean"]
    Proposed_REMBO_OTA_two_var = Proposed_REMBO_OTA_two["var"]
    REMBO_all_OTA_three_iter_times = REMBO_all_OTA_three["iter_times"]
    REMBO_all_OTA_three_mean = REMBO_all_OTA_three["mean"]
    REMBO_all_OTA_three_var = REMBO_all_OTA_three["var"]
    Proposed_REMBO_OTA_three_iter_times = Proposed_REMBO_OTA_three["iter_times"]
    Proposed_REMBO_OTA_three_mean = Proposed_REMBO_OTA_three["mean"]
    Proposed_REMBO_OTA_three_var = Proposed_REMBO_OTA_three["var"]
    REMBO_all_bandgap_iter_times = REMBO_all_bandgap["iter_times"]
    REMBO_all_bandgap_mean = REMBO_all_bandgap["mean"]
    REMBO_all_bandgap_var = REMBO_all_bandgap["var"]
    Proposed_REMBO_bandgap_iter_times = Proposed_REMBO_bandgap["iter_times"]
    Proposed_REMBO_bandgap_mean = Proposed_REMBO_bandgap["mean"]
    Proposed_REMBO_bandgap_var = Proposed_REMBO_bandgap["var"]
    Proposed_SMAC_OTA_two_iter_times = Proposed_SMAC_OTA_two["iter_times"]
    Proposed_SMAC_OTA_two_mean = Proposed_SMAC_OTA_two["mean"]
    Proposed_SMAC_OTA_two_var = Proposed_SMAC_OTA_two["var"]
    Proposed_SMAC_OTA_three_iter_times = Proposed_SMAC_OTA_three["iter_times"]
    Proposed_SMAC_OTA_three_mean = Proposed_SMAC_OTA_three["mean"]
    Proposed_SMAC_OTA_three_var = Proposed_SMAC_OTA_three["var"]
    Proposed_SMAC_bandgap_iter_times = Proposed_SMAC_bandgap["iter_times"]
    Proposed_SMAC_bandgap_mean = Proposed_SMAC_bandgap["mean"]
    Proposed_SMAC_bandgap_var = Proposed_SMAC_bandgap["var"]
    USeMOC_all_OTA_two_iter_times = USeMOC_all_OTA_two["iter_times"]
    USeMOC_all_OTA_two_mean = USeMOC_all_OTA_two["mean"]
    USeMOC_all_OTA_two_var = USeMOC_all_OTA_two["var"]
    Proposed_USeMOC_OTA_two_iter_times = Proposed_USeMOC_OTA_two["iter_times"]
    Proposed_USeMOC_OTA_two_mean = Proposed_USeMOC_OTA_two["mean"]
    Proposed_USeMOC_OTA_two_var = Proposed_USeMOC_OTA_two["var"]
    USeMOC_all_OTA_three_iter_times = USeMOC_all_OTA_three["iter_times"]
    USeMOC_all_OTA_three_mean = USeMOC_all_OTA_three["mean"]
    USeMOC_all_OTA_three_var = USeMOC_all_OTA_three["var"]
    Proposed_USeMOC_OTA_three_iter_times = Proposed_USeMOC_OTA_three["iter_times"]
    Proposed_USeMOC_OTA_three_mean = Proposed_USeMOC_OTA_three["mean"]
    Proposed_USeMOC_OTA_three_var = Proposed_USeMOC_OTA_three["var"]
    USeMOC_all_bandgap_iter_times = USeMOC_all_bandgap["iter_times"]
    USeMOC_all_bandgap_mean = USeMOC_all_bandgap["mean"]
    USeMOC_all_bandgap_var = USeMOC_all_bandgap["var"]
    Proposed_USeMOC_bandgap_iter_times = Proposed_USeMOC_bandgap["iter_times"]
    Proposed_USeMOC_bandgap_mean = Proposed_USeMOC_bandgap["mean"]
    Proposed_USeMOC_bandgap_var = Proposed_USeMOC_bandgap["var"]

    ratio = 0.5
    fig, ax = plt.subplots()
    for i in range(1):
        # 二级proposed_传统BO VS 传统BO
        if task == 0:
            ratio = 0.3
            plt.plot(Unbind_all_OTA_two_iter_times, Unbind_all_OTA_two_mean, linewidth=1.5, color=color[1],
                     label='BO', linestyle='-.')
            plt.fill_between(Unbind_all_OTA_two_iter_times, Unbind_all_OTA_two_mean - Unbind_all_OTA_two_var * ratio,
                             Unbind_all_OTA_two_mean + Unbind_all_OTA_two_var * ratio, alpha=0.2, color=color[1])
            plt.plot(Proposed_OTA_two_iter_times, Proposed_OTA_two_mean, linewidth=3, color=color[0], label='BO+DSFold')
            plt.fill_between(Proposed_OTA_two_iter_times, Proposed_OTA_two_mean - Proposed_OTA_two_var * ratio,
                             Proposed_OTA_two_mean + Proposed_OTA_two_var * ratio, alpha=0.2, color=color[0])

        # 三级proposed_传统BO VS 传统BO
        elif task == 1:
            ratio = 0.3
            plt.plot(Unbind_all_OTA_three_iter_times, Unbind_all_OTA_three_mean, linewidth=1.5, color=color[1],
                     label='BO',
                     linestyle='-.')
            plt.fill_between(Unbind_all_OTA_three_iter_times,
                             Unbind_all_OTA_three_mean - Unbind_all_OTA_three_var * ratio,
                             Unbind_all_OTA_three_mean + Unbind_all_OTA_three_var * ratio, alpha=0.2, color=color[1])

            plt.plot(Proposed_OTA_three_iter_times, Proposed_OTA_three_mean, linewidth=3, color=color[0],
                     label='BO+DSFold',  zorder=2)
            plt.fill_between(Proposed_OTA_three_iter_times, Proposed_OTA_three_mean - Proposed_OTA_three_var * ratio,
                             Proposed_OTA_three_mean + Proposed_OTA_three_var * ratio, alpha=0.2, color=color[0],  zorder=2)

        # 二级proposed_MACE VS 传统MACE
        elif task == 2:
            plt.plot(MACE_all_OTA_two_iter_times, MACE_all_OTA_two_mean, linewidth=1.5, color=color[2],
                     label='MACE', linestyle='-.')
            plt.fill_between(MACE_all_OTA_two_iter_times, MACE_all_OTA_two_mean - MACE_all_OTA_two_var * ratio,
                             MACE_all_OTA_two_mean + MACE_all_OTA_two_var * ratio, alpha=0.2, color=color[2])
            plt.plot(Proposed_MACE_OTA_two_iter_times, Proposed_MACE_OTA_two_mean, linewidth=3, color=color[0],
                     label='MACE+DSFold')
            plt.fill_between(Proposed_MACE_OTA_two_iter_times, Proposed_MACE_OTA_two_mean - Proposed_MACE_OTA_two_var * ratio,
                             Proposed_MACE_OTA_two_mean + Proposed_MACE_OTA_two_var * ratio, alpha=0.2, color=color[0])

        # 三级proposed_MACE VS 传统MACE
        elif task == 3:
            plt.plot(MACE_all_OTA_three_iter_times, MACE_all_OTA_three_mean, linewidth=1.5, color=color[2],
                     label='MACE', linestyle='-.')
            plt.fill_between(MACE_all_OTA_three_iter_times, MACE_all_OTA_three_mean - MACE_all_OTA_three_var * ratio,
                             MACE_all_OTA_three_mean + MACE_all_OTA_three_var * ratio, alpha=0.2, color=color[2])
            plt.plot(Proposed_MACE_OTA_three_iter_times, Proposed_MACE_OTA_three_mean, linewidth=3, color=color[0],
                     label='MACE+DSFold')
            plt.fill_between(Proposed_MACE_OTA_three_iter_times,
                             Proposed_MACE_OTA_three_mean - Proposed_MACE_OTA_three_var * ratio,
                             Proposed_MACE_OTA_three_mean + Proposed_MACE_OTA_three_var * ratio, alpha=0.2, color=color[0])

        # 二级proposed_REMBO VS 传统REMBO
        elif task == 4:
            plt.plot(REMBO_all_OTA_two_iter_times, REMBO_all_OTA_two_mean, linewidth=1.5, color=color[3],
                     label='REMBO', linestyle='-.')
            plt.fill_between(REMBO_all_OTA_two_iter_times, REMBO_all_OTA_two_mean - REMBO_all_OTA_two_var * ratio,
                             REMBO_all_OTA_two_mean + REMBO_all_OTA_two_var * ratio, alpha=0.2, color=color[3])
            plt.plot(Proposed_REMBO_OTA_two_iter_times, Proposed_REMBO_OTA_two_mean, linewidth=3, color=color[0],
                     label='REMBO+DSFold')
            plt.fill_between(Proposed_REMBO_OTA_two_iter_times,
                             Proposed_REMBO_OTA_two_mean - Proposed_REMBO_OTA_two_var * ratio,
                             Proposed_REMBO_OTA_two_mean + Proposed_REMBO_OTA_two_var * ratio, alpha=0.2, color=color[0])

        # 三级proposed_REMBO VS 传统REMBO
        elif task == 5:
            plt.plot(REMBO_all_OTA_three_iter_times, REMBO_all_OTA_three_mean, linewidth=1.5, color=color[3],
                     label='REMBO', linestyle='-.')
            plt.fill_between(REMBO_all_OTA_three_iter_times, REMBO_all_OTA_three_mean - REMBO_all_OTA_three_var * ratio,
                             REMBO_all_OTA_three_mean + REMBO_all_OTA_three_var * ratio, alpha=0.2, color=color[3])
            plt.plot(Proposed_REMBO_OTA_three_iter_times, Proposed_REMBO_OTA_three_mean, linewidth=3, color=color[0],
                     label='REMBO+DSFold')
            plt.fill_between(Proposed_REMBO_OTA_three_iter_times,
                             Proposed_REMBO_OTA_three_mean - Proposed_REMBO_OTA_three_var * ratio,
                             Proposed_REMBO_OTA_three_mean + Proposed_REMBO_OTA_three_var * ratio, alpha=0.2, color=color[0])

        # 二级Proposed_SMAC VS 传统SMAC
        elif task == 6:
            plt.plot(SMAC_all_OTA_two_iter_times, SMAC_all_OTA_two_mean, linewidth=1.5, color=color[6],
                     label='SMAC', linestyle='-.')
            plt.fill_between(SMAC_all_OTA_two_iter_times, SMAC_all_OTA_two_mean - SMAC_all_OTA_two_var * ratio,
                             SMAC_all_OTA_two_mean + SMAC_all_OTA_two_var * ratio, alpha=0.2, color=color[6])
            plt.plot(Proposed_SMAC_OTA_two_iter_times, Proposed_SMAC_OTA_two_mean, linewidth=3, color=color[0],
                     label='SMAC+DSFold')
            plt.fill_between(Proposed_SMAC_OTA_two_iter_times,
                             Proposed_SMAC_OTA_two_mean - Proposed_SMAC_OTA_two_var * ratio,
                             Proposed_SMAC_OTA_two_mean + Proposed_SMAC_OTA_two_var * ratio, alpha=0.2, color=color[0])

        # 三级Proposed_SMAC VS 传统SMAC
        elif task == 7:
            plt.plot(SMAC_all_OTA_three_iter_times, SMAC_all_OTA_three_mean, linewidth=1.5, color=color[6],
                     label='SMAC', linestyle='-.')
            plt.fill_between(SMAC_all_OTA_three_iter_times, SMAC_all_OTA_three_mean - SMAC_all_OTA_three_var * ratio,
                             SMAC_all_OTA_three_mean + SMAC_all_OTA_three_var * ratio, alpha=0.2, color=color[6])
            plt.plot(Proposed_SMAC_OTA_three_iter_times, Proposed_SMAC_OTA_three_mean, linewidth=3, color=color[0],
                     label='SMAC+DSFold')
            plt.fill_between(Proposed_SMAC_OTA_three_iter_times,
                             Proposed_SMAC_OTA_three_mean - Proposed_SMAC_OTA_three_var * ratio,
                             Proposed_SMAC_OTA_three_mean + Proposed_SMAC_OTA_three_var * ratio, alpha=0.2, color=color[0])

        # 二级Proposed_USeMOC VS 传统USeMOC
        elif task == 8:
            plt.plot(USeMOC_all_OTA_two_iter_times, USeMOC_all_OTA_two_mean, linewidth=1.5, color=color[4],
                     label='USeMOC', linestyle='-.')
            plt.fill_between(USeMOC_all_OTA_two_iter_times, USeMOC_all_OTA_two_mean - USeMOC_all_OTA_two_var * ratio,
                             USeMOC_all_OTA_two_mean + USeMOC_all_OTA_two_var * ratio, alpha=0.2, color=color[4])
            plt.plot(Proposed_USeMOC_OTA_two_iter_times, Proposed_USeMOC_OTA_two_mean, linewidth=3, color=color[0],
                     label='USeMOC+DSFold')
            plt.fill_between(Proposed_USeMOC_OTA_two_iter_times,
                             Proposed_USeMOC_OTA_two_mean - Proposed_USeMOC_OTA_two_var * ratio,
                             Proposed_USeMOC_OTA_two_mean + Proposed_USeMOC_OTA_two_var * ratio, alpha=0.2, color=color[0])

        # 三级Proposed_USeMOC VS 传统USeMOC
        elif task == 9:
            plt.plot(USeMOC_all_OTA_three_iter_times, USeMOC_all_OTA_three_mean, linewidth=1.5, color=color[4],
                     label='USeMOC', linestyle='-.')
            plt.fill_between(USeMOC_all_OTA_three_iter_times, USeMOC_all_OTA_three_mean - USeMOC_all_OTA_three_var * ratio,
                             USeMOC_all_OTA_three_mean + USeMOC_all_OTA_three_var * ratio, alpha=0.2, color=color[4])
            plt.plot(Proposed_USeMOC_OTA_three_iter_times, Proposed_USeMOC_OTA_three_mean, linewidth=3, color=color[0],
                     label='USeMOC+DSFold')
            plt.fill_between(Proposed_USeMOC_OTA_three_iter_times,
                             Proposed_USeMOC_OTA_three_mean - Proposed_USeMOC_OTA_three_var * ratio,
                             Proposed_USeMOC_OTA_three_mean + Proposed_USeMOC_OTA_three_var * ratio, alpha=0.2,
                             color=color[0])

        # 二级Proposed_BO VS 传统SMAC VS 传统USeMOC
        elif task == 10:
            ratio = 0.3
            plt.plot(USeMOC_all_OTA_two_iter_times, USeMOC_all_OTA_two_mean, linewidth=1.5, color=color[4],
                     label='USeMOC', linestyle='-.')
            plt.fill_between(USeMOC_all_OTA_two_iter_times, USeMOC_all_OTA_two_mean - USeMOC_all_OTA_two_var * ratio,
                             USeMOC_all_OTA_two_mean + USeMOC_all_OTA_two_var * ratio, alpha=0.2, color=color[4])
            plt.plot(SMAC_all_OTA_two_iter_times, SMAC_all_OTA_two_mean, linewidth=1.5, color=color[6],
                     label='SMAC', linestyle='--')
            plt.fill_between(SMAC_all_OTA_two_iter_times, SMAC_all_OTA_two_mean - SMAC_all_OTA_two_var * ratio,
                             SMAC_all_OTA_two_mean + SMAC_all_OTA_two_var * ratio, alpha=0.2, color=color[6])
            plt.plot(Proposed_OTA_two_iter_times, Proposed_OTA_two_mean, linewidth=3, color=color[0],
                     label='BO+DSFold')
            plt.fill_between(Proposed_OTA_two_iter_times, Proposed_OTA_two_mean - Proposed_OTA_two_var * ratio,
                             Proposed_OTA_two_mean + Proposed_OTA_two_var * ratio, alpha=0.2, color=color[0])

        # 三级Proposed_BO VS 传统SMAC VS 传统USeMOC
        elif task == 11:
            plt.plot(USeMOC_all_OTA_three_iter_times, USeMOC_all_OTA_three_mean, linewidth=1.5, color=color[4],
                     label='USeMOC', linestyle='-.')
            plt.fill_between(USeMOC_all_OTA_three_iter_times,
                             USeMOC_all_OTA_three_mean - USeMOC_all_OTA_three_var * ratio,
                             USeMOC_all_OTA_three_mean + USeMOC_all_OTA_three_var * ratio, alpha=0.2, color=color[4])
            plt.plot(SMAC_all_OTA_three_iter_times, SMAC_all_OTA_three_mean, linewidth=1.5, color=color[6],
                     label='SMAC', linestyle='--')
            plt.fill_between(SMAC_all_OTA_three_iter_times, SMAC_all_OTA_three_mean - SMAC_all_OTA_three_var * ratio,
                             SMAC_all_OTA_three_mean + SMAC_all_OTA_three_var * ratio, alpha=0.2, color=color[6])
            plt.plot(Proposed_OTA_three_iter_times, Proposed_OTA_three_mean, linewidth=3, color=color[0],
                     label='BO+DSFold')
            plt.fill_between(Proposed_OTA_three_iter_times, Proposed_OTA_three_mean - Proposed_OTA_three_var * ratio,
                             Proposed_OTA_three_mean + Proposed_OTA_three_var * ratio, alpha=0.2, color=color[0])

        # bandgap proposed_传统BO VS 传统BO
        if task == 12:
            plt.plot(Unbind_all_bandgap_iter_times, Unbind_all_bandgap_mean, linewidth=1.5, color=color[1],
                     label='BO', linestyle='-.')
            plt.fill_between(Unbind_all_bandgap_iter_times,
                             Unbind_all_bandgap_mean - Unbind_all_bandgap_var * ratio,
                             Unbind_all_bandgap_mean + Unbind_all_bandgap_var * ratio, alpha=0.2, color=color[1])
            plt.plot(Proposed_bandgap_iter_times, Proposed_bandgap_mean, linewidth=3, color=color[0],
                     label='BO+DSFold')
            plt.fill_between(Proposed_bandgap_iter_times, Proposed_bandgap_mean - Proposed_bandgap_var * ratio,
                             Proposed_bandgap_mean + Proposed_bandgap_var * ratio, alpha=0.2, color=color[0])

        # bandgap proposed_MACE VS 传统MACE
        elif task == 13:
            plt.plot(MACE_all_bandgap_iter_times, MACE_all_bandgap_mean, linewidth=1.5, color=color[2],
                     label='MACE', linestyle='-.')
            plt.fill_between(MACE_all_bandgap_iter_times, MACE_all_bandgap_mean - MACE_all_bandgap_var * ratio,
                             MACE_all_bandgap_mean + MACE_all_bandgap_var * ratio, alpha=0.2, color=color[2])
            plt.plot(Proposed_MACE_bandgap_iter_times, Proposed_MACE_bandgap_mean, linewidth=3, color=color[0],
                     label='MACE+DSFold')
            plt.fill_between(Proposed_MACE_bandgap_iter_times,
                             Proposed_MACE_bandgap_mean - Proposed_MACE_bandgap_var * ratio,
                             Proposed_MACE_bandgap_mean + Proposed_MACE_bandgap_var * ratio, alpha=0.2,
                             color=color[0])

        # bandgap proposed_REMBO VS 传统REMBO
        elif task == 14:
            ratio = 0.3
            plt.plot(REMBO_all_bandgap_iter_times, REMBO_all_bandgap_mean, linewidth=1.5, color=color[3],
                     label='REMBO', linestyle='-.')
            plt.fill_between(REMBO_all_bandgap_iter_times, REMBO_all_bandgap_mean - REMBO_all_bandgap_var * ratio,
                             REMBO_all_bandgap_mean + REMBO_all_bandgap_var * ratio, alpha=0.2, color=color[3])
            plt.plot(Proposed_REMBO_bandgap_iter_times, Proposed_REMBO_bandgap_mean, linewidth=3, color=color[0],
                     label='REMBO+DSFold')
            plt.fill_between(Proposed_REMBO_bandgap_iter_times,
                             Proposed_REMBO_bandgap_mean - Proposed_REMBO_bandgap_var * ratio,
                             Proposed_REMBO_bandgap_mean + Proposed_REMBO_bandgap_var * ratio, alpha=0.2,
                             color=color[0])

        # bandgap Proposed_SMAC VS 传统SMAC
        elif task == 15:
            ratio = 0.1
            plt.plot(SMAC_all_bandgap_iter_times, SMAC_all_bandgap_mean, linewidth=1.5, color=color[6],
                     label='SMAC', linestyle='-.')
            plt.fill_between(SMAC_all_bandgap_iter_times, SMAC_all_bandgap_mean - SMAC_all_bandgap_var * ratio,
                             SMAC_all_bandgap_mean + SMAC_all_bandgap_var * ratio, alpha=0.2, color=color[6])
            plt.plot(Proposed_SMAC_bandgap_iter_times, Proposed_SMAC_bandgap_mean, linewidth=3, color=color[0],
                     label='SMAC+DSFold')
            plt.fill_between(Proposed_SMAC_bandgap_iter_times,
                             Proposed_SMAC_bandgap_mean - Proposed_SMAC_bandgap_var * ratio,
                             Proposed_SMAC_bandgap_mean + Proposed_SMAC_bandgap_var * ratio, alpha=0.2,
                             color=color[0])

        # bandgap Proposed_USeMOC VS 传统USeMOC
        elif task == 16:
            plt.plot(USeMOC_all_bandgap_iter_times, USeMOC_all_bandgap_mean, linewidth=1.5, color=color[4],
                     label='USeMOC', linestyle='-.')
            plt.fill_between(USeMOC_all_bandgap_iter_times,
                             USeMOC_all_bandgap_mean - USeMOC_all_bandgap_var * ratio,
                             USeMOC_all_bandgap_mean + USeMOC_all_bandgap_var * ratio, alpha=0.2, color=color[4])
            plt.plot(Proposed_USeMOC_bandgap_iter_times, Proposed_USeMOC_bandgap_mean, linewidth=3, color=color[0],
                     label='USeMOC+DSFold')
            plt.fill_between(Proposed_USeMOC_bandgap_iter_times,
                             Proposed_USeMOC_bandgap_mean - Proposed_USeMOC_bandgap_var * ratio,
                             Proposed_USeMOC_bandgap_mean + Proposed_USeMOC_bandgap_var * ratio, alpha=0.2,
                             color=color[0])

        # bandgap Proposed_BO VS 传统SMAC VS 传统USeMOC
        elif task == 17:
            ratio = 0.3
            plt.plot(USeMOC_all_bandgap_iter_times, USeMOC_all_bandgap_mean, linewidth=1.5, color=color[4],
                     label='USeMOC', linestyle='-.')
            plt.fill_between(USeMOC_all_bandgap_iter_times,
                             USeMOC_all_bandgap_mean - USeMOC_all_bandgap_var * ratio,
                             USeMOC_all_bandgap_mean + USeMOC_all_bandgap_var * ratio, alpha=0.2, color=color[4])
            plt.plot(SMAC_all_bandgap_iter_times, SMAC_all_bandgap_mean, linewidth=1.5, color=color[6],
                     label='SMAC', linestyle='--')
            plt.fill_between(SMAC_all_bandgap_iter_times, SMAC_all_bandgap_mean - SMAC_all_bandgap_var * ratio,
                             SMAC_all_bandgap_mean + SMAC_all_bandgap_var * ratio, alpha=0.2, color=color[6])
            plt.plot(Proposed_bandgap_iter_times, Proposed_bandgap_mean, linewidth=3, color=color[0],
                     label='BO+DSFold')
            plt.fill_between(Proposed_bandgap_iter_times, Proposed_bandgap_mean - Proposed_bandgap_var * ratio,
                             Proposed_bandgap_mean + Proposed_bandgap_var * ratio, alpha=0.2, color=color[0])

        # 二级电路可行点搜索箱线图
        elif task == 18:
            RS_all = []
            BO_all = []
            NSGAII_all = []
            MACE_all = []
            RS_bind = [58, 9, 37, 4, 39]
            BO_bind = []
            NSGAII_bind = []
            MACE_bind = []

            box_color = ['#8c564b',     # 棕色         RS用
                         '#ff7f0e',     # 橙色         传统BO用
                         '#17becf',     # 浅蓝色        MSGAII
                         '#2ca02c',     # 绿色         MACE
                         ]

            # 绘制箱线图
            plt.boxplot([RS_all, BO_all, NSGAII_all, MACE_all, RS_bind, BO_bind,NSGAII_bind, MACE_bind], patch_artist=True,
                        boxprops=dict(facecolor=box_color))

            # 添加图例
            plt.legend(['RS_all', 'BO_all', 'NSGAII_all', 'MACE_all', 'RS_bind', 'BO_bind', 'NSGAII_bind', 'MACE_bind'])

            # 添加标题和标签
            plt.title('Multiple Box Plots')
            plt.xlabel('Groups')
            plt.ylabel('Values')

            # 自定义x轴刻度标签
            plt.xticks([1, 2, 3], ['Group 1', 'Group 2', 'Group 3'])

            # 显示图形
            plt.show()



    # 标题，横纵标签，字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
    if 12 <= task <= 17:       # ppm
        # plt.xlim(np.arange(10, 105, 10))
        # 获取x轴的当前范围
        # x_start, x_end = plt.gca().get_xlim()

        # 设置x轴的刻度步长
        # plt.xticks(np.arange(10, 105, 10))
        # plt.title("three-Stage Operational Amplifier", fontsize=15)
        plt.ylabel(y_label[0], fontsize=25, fontname='Times New Roman')
    elif 0 <= task <= 11:                  # Itotal
        # plt.title("CON_BO_Result", fontsize=15)

        plt.gca().ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_visible(False)
        # 创建刻度格式化对象并应用于y轴
        scalar_formatter = ScalarFormatter()
        scalar_formatter.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(scalar_formatter)

        # 将刻度值除以10
        def divide_by_10(x, pos):
            return f'{x*1000:.2}'

        ax.yaxis.set_major_formatter(plt.FuncFormatter(divide_by_10))
        plt.ylabel(y_label[4], fontsize=25, fontname='Times New Roman')

    plt.xlabel("Number of Simulations", fontsize=25, fontname='Times New Roman')



    # 设置刻度标记大小tick_params()
    ax = plt.gca()
    plt.tick_params(axis='both', labelsize=18)
    if task == 20:
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.legend(loc='lower right', fontsize=20.5)  # 添加图例 右下
    if 0 <= task <= 17:
        plt.rcParams['font.family'] = 'Times New Roman'

        # if task == 1:
        #     # 假设这是你的x和y值列表
        #     x_values = np.array(Proposed_OTA_three_iter_times)
        #     y_values = np.array(Proposed_OTA_three_mean)
        #
        #     # 假设星星要添加到以下x值上
        #     x_stars = [70, 136, 173, 245, 275]
        #
        #     # 计算红色曲线上的y值
        #     y_stars = np.interp(x_stars, x_values, y_values)
        #
        #     # 在红色曲线上的特定点绘制星星
        #     plt.scatter(x_stars, y_stars, color='darkblue', marker='*', s=150, zorder=3, label='Design Space Expansion')  # s为星星的大小
        plt.legend(loc='upper right', fontsize=20.5)  # 添加图例 右上
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    value = [8, 9, 16]
    for i in value:
        plot_first_strand(i)







