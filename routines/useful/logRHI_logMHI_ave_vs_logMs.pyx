#################################################################################
#                                   VARIABLES
#################################################################################

def ave_logRHI_logMs():
    from distutils.core import setup
    from Cython.Build import cythonize
    import os.path #needed for paths
    import numpy as np
    
    setup(
    ext_modules = cythonize("module_gas.pyx")
    )
    
    import module_gas
    
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import matplotlib.font_manager
    from matplotlib.ticker import ScalarFormatter
    from os import listdir
    
    from matplotlib.colors import LogNorm
    from scipy.stats import gaussian_kde
    
    import matplotlib.cm as cm
    from matplotlib import rc
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
    

    
    #(1): conditional distributions parameters:
    HI_LTG_params = [0,-0.127167,1.27935,2.59764,8.64563,-0.0180047,0.576922]
    H2_LTG_params = [0,-0.0850781,0.830319,0.121454,10.5945,0.840925,0.062628]
    HI_ETG_params = [0,-0.0519849,-0.0741485,1.57328,8.35398,-0.819537,0.46762,0.0602505,-0.112526,-0.258828,-0.309552]
    H2_ETG_params = [0,0.0585967,-1.49083,0.673683,8.18159,-0.685737,0.375225,0.0174391,0.515328,-1.08394,7.98043]
    
    #(2):
    logRHI_all_min=-8.0
    logRHI_all_max=3.0
    logRH2_all_min=-8.0
    logRH2_all_max=3.0
    logRHI_ETGs_min=-8.0
    logRHI_ETGs_max=3.0
    logRH2_ETGs_min=-8.0
    logRH2_ETGs_max=3.0
    logRgas_LTGs_min=-8.0
    logRgas_LTGs_max=3.0
    logRHI_LTGs_min=-8.0
    logRHI_LTGs_max=3.0
    
    #===================================================================================#
    NbinsMs = 20;
    logMs_min = 7.0;
    logMs_max = 12.0;
    
    DeltaMs = (logMs_max - logMs_min) / NbinsMs;
    
    NbinsMHj = 400;
    logRHj_min = -6.5;
    logRHj_max = 2.5;
    logMHj_min = logRHj_min + logMs_min;
    logMHj_max = logRHj_max + logMs_max;
    
    DeltaMHj = (logMHj_max - logMHj_min) / NbinsMHj;
    
    logMs = []
    logRHj_ave_LTGs = []
    SD_logRHj_LTGs = []
    logRHj_ave_ETGs = []
    SD_logRHj_ETGs = []
    logRHj_ave_TOT = []
    SD_logRHj_TOT = []
    results = {}

    for i in range(NbinsMs + 1):
        logMs_x = logMs_min + i * DeltaMs
    
        logRHj_ave_x_LTGs = module_gas.FM_PHj_LTGs(logMs_x, HI_LTG_params, logRHI_LTGs_min, logRHI_LTGs_max, 1.0)
        SD_logRHj_x_LTGs = module_gas.SM_PHj_LTGs(logMs_x, HI_LTG_params, logRHI_LTGs_min, logRHI_LTGs_max, 1.0)
        SD_logRHj_x_LTGs = np.sqrt(SD_logRHj_x_LTGs)
    
        logRHj_ave_x_ETGs = module_gas.FM_PHj_ETGs(logMs_x, HI_ETG_params, logRHI_ETGs_min, logRHI_ETGs_max, 1.0)
        SD_logRHj_x_ETGs = module_gas.SM_PHj_ETGs(logMs_x, HI_ETG_params, logRHI_ETGs_min, logRHI_ETGs_max, 1.0)
        SD_logRHj_x_ETGs = np.sqrt(SD_logRHj_x_ETGs)
    
        logRHj_ave_x_TOT = module_gas.FirstMom_P_RHj_all(logMs_x, HI_LTG_params, HI_ETG_params, logRHI_all_min, logRHI_all_max, 1.0)
        SD_logRHj_x_TOT = module_gas.SecondMom_P_RHj_all(logMs_x, HI_LTG_params, HI_ETG_params, logRHI_all_min, logRHI_all_max, 1.0)
        SD_logRHj_x_TOT = np.sqrt(SD_logRHj_x_TOT)
    
        logMs.append(logMs_x)
    
        logRHj_ave_LTGs.append(logRHj_ave_x_LTGs)
        SD_logRHj_LTGs.append(SD_logRHj_x_LTGs)
    
        logRHj_ave_ETGs.append(logRHj_ave_x_ETGs)
        SD_logRHj_ETGs.append(SD_logRHj_x_ETGs)
    
        logRHj_ave_TOT.append(logRHj_ave_x_TOT)
        SD_logRHj_TOT.append(SD_logRHj_x_TOT)
    
    
    results['logMs'] = logMs
    results['logRHj_ave_LTGs'] = logRHj_ave_LTGs
    results['SD_logRHj_LTGs'] = SD_logRHj_LTGs
    results['logRHj_ave_ETGs'] = logRHj_ave_ETGs
    results['SD_logRHj_ETGs'] = SD_logRHj_ETGs
    results['logRHj_ave_TOT'] = logRHj_ave_TOT
    results['SD_logRHj_TOT'] = SD_logRHj_TOT
    
    print(list(results.keys()))
    print(results)
    # f.close()
    
    #################################################################################
    #                            Figures
    #################################################################################
    #Fonts
    rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
    rc('text', usetex=True)
    font_type = {'fontname':'Serif'}
    
    fig = plt.figure(figsize=None)
    
    #################################################################################
    ax = plt.subplot2grid((20,20), (4,4), rowspan=10, colspan=10)
    
    #----------------------------#
    ax.fill_between(results['logMs'], np.array(results['logRHj_ave_LTGs'])-np.array(results['SD_logRHj_LTGs']), np.array(results['logRHj_ave_LTGs'])+np.array(results['SD_logRHj_LTGs']), alpha=0.4, edgecolor='blue', facecolor='blue', linewidth=0.1)
    ax.plot(results['logMs'], results['logRHj_ave_LTGs'], color = 'blue', marker='.', linestyle='-', mec='none', markerfacecolor='blue', markersize='1.0', linewidth = 1.0, zorder = 1) #marker="None"
    
    ax.fill_between(results['logMs'], np.array(results['logRHj_ave_ETGs'])-np.array(results['SD_logRHj_ETGs']), np.array(results['logRHj_ave_ETGs'])+np.array(results['SD_logRHj_ETGs']), alpha=0.4, edgecolor='red', facecolor='red', linewidth=0.1)
    ax.plot(results['logMs'], results['logRHj_ave_ETGs'], color = 'red', marker='.', linestyle='-', mec='none', markerfacecolor='blue', markersize='1.0', linewidth = 1.0, zorder = 1) #marker="None"
    
    ax.fill_between(results['logMs'], np.array(results['logRHj_ave_TOT'])-np.array(results['SD_logRHj_TOT']), np.array(results['logRHj_ave_TOT'])+np.array(results['SD_logRHj_TOT']), alpha=0.4, edgecolor='black', facecolor='black', linewidth=0.1)
    ax.plot(results['logMs'], results['logRHj_ave_TOT'], color = 'black', marker='.', linestyle='-', mec='none', markerfacecolor='blue', markersize='1.0', linewidth = 1.0, zorder = 1) #marker="None"
    
    #----------------------------#
    #Hide or show values of x and y axis
    plt.setp(ax.get_yticklabels(), visible=True) #Hide values of x or y axis
    plt.setp(ax.get_xticklabels(), visible=True) #Hide values of x or y axis
    #Limits
    ax.set_xlim(7, 12.0)
    ax.set_ylim(-4.5, 2.5)
    #axis scale
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    #Change labels ticks size
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    #Labels
    ax.set_xlabel('$\\rm \log M_{\\ast}$  $\\rm [M_{\\odot}]$' , fontsize = '14')
    ax.set_ylabel('$\\langle \log {\\rm R_{HI}}\\rangle$' , fontsize = '14')
    #Ticks
    #ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1)) #Major ticks
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.2)) #Minor ticks
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1)) #Major ticks
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2)) #Minor ticks
    ax.tick_params(which='major', length=8, direction="in", top=True, right=True) #length of major ticks
    ax.tick_params(which='minor', length=4, direction="in", top=True, right=True) #length of minor ticks
    #Legend
    #ax.legend(frameon = False, loc = 'upper left', fontsize = '4')
    
    #################################################################################
    #title
    #titlefig = fig.suptitle('Using $\log(\\rm M_{\\ast})-\log (\\rm V_{max})$ and $\log(\\rm M_{\\ast})-\log (\\rm V_{peak})$ correlation for ETGs and LTGs', fontsize='10', fontweight='light', **font_type)
    #titlefig.set_y(0.88) #Where to place the title
    fig.subplots_adjust(top=0.92) #Adjust subplots with respect title
    
    #configuring subplots
    fig.subplots_adjust(left=0.1, bottom=None, right=0.99, top=None, wspace=0.0, hspace=0.0)
    #fig.set_tight_layout(True) !To automatically leave the subplots symmetric
    #plt.draw()
    #plt.show()
    plt.savefig('results.pdf', Transparent=True)
    
    ##################################################################################
    
