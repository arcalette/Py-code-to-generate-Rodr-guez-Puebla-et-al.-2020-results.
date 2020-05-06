# -*- encoding: utf-8 -*-
"""
Module: module1.py
Author: Ángel Rubén Calette Morín.

Description:
This is the module of main program coldgas.py and contains information regarding file paths,
names and some functions that are useful in the main program.

Important sources:

"""

# imports
from numpy import log, log10, exp, power
import numpy as np
import os.path #needed for paths
from scipy.special import gamma, gammaincc
import scipy.integrate as integrate

#################################################################################
#                                   VARIABLES
#################################################################################
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

#################################################################################
#                                     PATHS
#################################################################################

################################################################################
#                                     FUNCTIONS
#################################################################################

    # TOP HAT PARAMETERS (ETGs) #
#/=======================================================================!
# FUNCTION:  x2_P_TH_dlogR                                              #
#   This function is the upper limit in the integral of the TH function,#
#   or the lower limit of Schechter function.                           #
#                                                                       #
#  Notes:                                                               #
#   (1):                                                                #
#                                                                       #
#=======================================================================*/
def x2_P_TH_dlogR(logMs, slope, b):
   return (slope * logMs) + b


#/=======================================================================!
# FUNCTION:  F_P_TH_dlogR                                               #
#   This function is a stellar mass dependent TopHat functional form    #
#   integral used for ETGs (HI and H2)                                  #
#                                                                       #
#  Notes:                                                               #
#   (1):                                                                #
#                                                                       #
#=======================================================================*/
def F_P_TH_dlogR(logMs, slope, b):
    return (slope * logMs) + b


#OTHER PARAMETERS #
#/=======================================================================!
# FUNCTION:  logRstr                                                    #
#   This function is a stellar mass dependent DPL functional form       #
#   of the logRstr parameter for Schechter function (HI and H2)         #
#                                                                       #
#  Notes:                                                               #
#   (1): DPL relation                                                   #
#                                                                       #
#=======================================================================*/
def logRstr(logMs, C, logMbr, a, b):
    ratio = np.power(10, logMs-logMbr)
    Term1 = np.power(ratio, a)
    Term2 = np.power(ratio, b)
    return np.log10(C) - np.log10(Term1 + Term2)


#/=======================================================================!
# FUNCTION:  alpha_logMs                                                #
#   This function is a stellar mass dependent linear functional form    #
#   of the alpha parameter for Schechter function (HI and H2)           #
#                                                                       #
#  Notes:                                                               #
#   (1): Linear relation                                                #
#                                                                       #
#=======================================================================*/
def alpha_logMs(logMs, slope, b):
    return (slope*logMs) + b


#/=======================================================================!
# FUNCTION:  phistr                                                #
#   This function is a stellar mass dependent  functional form          #
#   of the phiStr parameter for Schechter function (HI and H2) for LTGs #
#                                                                       #
#  Notes:                                                               #
#   (1): Remember in this case phiStr = 1 / Gamma(1 + alpha)            #
#   (2): alpha = Schechter slope parameter.                             #
#   (3): phiStr = Built in such way that integral of Schechter is 1.    #
#                                                                       #
#=======================================================================*/
def phistr(alpha):
    return 1.0 / gamma(1.0 + alpha)


# DISTRIBUTIONS AND DISTRIBUTIONS INTEGRALS #
#/=======================================================================!
# FUNCTION:  Sx_logx                                                    #
#                                                                       #
#  Notes:                                                               #
#   (1):                                                                #
#                                                                       #
#=======================================================================*/
def Sx_logx(logRHj, alpha, logRHj_char):
    ratio = np.power(10, logRHj - logRHj_char)
    return  np.power(ratio, alpha) * np.exp(-ratio) * ratio * np.log(10.0)

#/=======================================================================!
# FUNCTION:  Sx_integral                                                #
#                                                                       #
# This function computes the the normalization of phi parameter for ETGs#
#                                                                       #
# Notes:                                                                #
#   (1):                                                                #
#                                                                       #
#=======================================================================*/
def Int_Sx_logx(alpha, logRst, logRHj_min, logRHj_max):
    N=50
    s = (N+1)*[0]
    integral=0.0
    integral1=0.0
    logRmin = logRHj_min
    logRmax = logRHj_max
    delta_logRHj = (logRmax - logRmin) / N

    # Integrating function #
    for i in range(N+1):
        logRHj_x = logRmin + ( i * delta_logRHj)
        PDF_MHj = Sx_logx(logRHj_x, alpha, logRst)
        s[i] = PDF_MHj
        if (i>=1 and i<=N-1):
            integral1+=s[i]

    integral =  0.5 * delta_logRHj * (s[0] + (2 * integral1) + s[N])
    return integral

#/=======================================================================!
# FUNCTION:  Schech_RHj_lin                                             #
#                                                                       #
#  Notes:                                                               #
#   (1):                                                                #
#                                                                       #
#=======================================================================*/
def Schech_RHj_lin(logRHj, alpha, logRHj_char):
    ratio = np.power(10, logRHj - logRHj_char)
    phist = phistr(alpha)
    return phist * np.power(ratio, alpha) * np.exp(-ratio)


#/=======================================================================!
# FUNCTION:  Schech_RHj_log                                             #
#                                                                       #
#  Notes:                                                               #
#   (1):                                                                #
#                                                                       #
#=======================================================================*/
def Schech_RHj_log(logRHj, alpha, logRHj_char):
    ratio = np.power(10, logRHj - logRHj_char)
    phist = phistr(alpha)
    return  Schech_RHj_lin(logRHj, alpha, logRHj_char) * ratio * np.log(10.0)


#=======================================================================!
# FUNCTION:  P_MHj_Ms_blue_LTGs                                         #
#                                                                       #
#  Notes:                                                               #
#   (1):                                                                #
#                                                                       #
#=======================================================================*/
def P_MHj_Ms_blue_LTGs(logMHj, logMs, params, flag_Hj):
 logRHj = logMHj - logMs

 if (flag_Hj == 1):  #HI
     logMs_x = logMs
 else: #H2
     logMs_x = 8.2 if logMs < 8.2 else logMs

 alpha = alpha_logMs(logMs_x, params[1], params[2])
 logRtr = logRstr(logMs_x, params[3], params[4], params[5], params[6])
 Int_den = gamma(1 + alpha)  if 1 + alpha > 0 else Int_Sx_logx(alpha, logRtr, -30.0, logMs_x)

 PDF_RHj = Sx_logx(logRHj, alpha, logRtr) / Int_den
 return PDF_RHj


#=======================================================================!
# FUNCTION:  P_MHj_Ms_red_ETGs                                          #
#                                                                       #
#  Notes:                                                               #
#   (1):                                                                #
#                                                                       #
#=======================================================================*/
def P_MHj_Ms_red_ETGs(logMHj, logMs, params, flag_Hj):
    width_P_TH=1.0
    logRHj = logMHj - logMs

    if (flag_Hj == 1): #HI
        logMs_x = 8.0 if logMs < 8.0 else logMs
    else:  #H2
        logMs_x = 9.0 if logMs < 9.0 else logMs

    # TopHat parameters */
    PDF_RHj_TH = F_P_TH_dlogR(logMs_x, params[7], params[8])
    x2_P_TH = x2_P_TH_dlogR(logMs_x, params[9], params[10])
    x1_P_TH = x2_P_TH - width_P_TH

    # Schechter parameters */
    alpha = alpha_logMs(logMs_x, params[1], params[2])
    logRtr = logRstr(logMs_x, params[3], params[4], params[5], params[6])
    x = np.power(10, x2_P_TH - logRtr)

    Gamma = gamma(1 + alpha) if 1 + alpha > 0 else Int_Sx_logx(alpha, logRtr, -30.0, 4.0)
    Q, err = gammq_err(1 + alpha, x)
    Int_den = Q * Gamma if err == 0 else  Int_Sx_logx(alpha, logRtr, x2_P_TH, 4.0)

    # Use Schechter or TopHat? */
    PDF_RHj = Sx_logx(logRHj, alpha, logRtr) * ( (1.0 - PDF_RHj_TH) / Int_den) if Int_den >= 0 else 0
    PDF_RHj = PDF_RHj_TH if logRHj <= x2_P_TH and logRHj >= x1_P_TH else PDF_RHj
    PDF_RHj = 0 if logRHj <= x1_P_TH else PDF_RHj
    return PDF_RHj

#=======================================================================!
# FUNCTION:  P_MHj_Ms_all                                               #
#                                                                       #
#  Notes:                                                               #
#   (1):                                                                #
#                                                                       #
#=======================================================================*/
def P_MHj_Ms_all(logMHj, logMs, params_LT, params_ET, flag_Hj):
    logRHj = logMHj - logMs

    PDF_RHj_LT = (1.0 - frac_e(logMs)) * P_MHj_Ms_blue_LTGs(logRHj + logMs, logMs, params_LT, flag_Hj)
    PDF_RHj_ET = frac_e(logMs) * P_MHj_Ms_red_ETGs(logRHj + logMs, logMs, params_ET, flag_Hj)
    PDF_RHj = PDF_RHj_LT + PDF_RHj_ET

    return PDF_RHj

#=======================================================================!
# FUNCTION:  gammq_err                                                  !
#                                                                       !
#                                                                       !
#                                                                       !
#  Notes:                                                               !
#  (1):                                                                 !
#=======================================================================*/
def gammq_err(a, x):
    err = 0.0
    gamm = 0.0

    if (x < 0.0 or a <= 0.0):
     err = 1.0
    else:
     gamm = gammaincc(a,x)

    return gamm, err

#========================================================================!
# FUNCTION:   frac_e                                                     #
#   This is the parametrization of the fraction of early galaxies        #
#   determined by Rodríguez-Puebla et al.                                #
#                                                                        #
#                                                                        #
#  Notes:                                                                #
#   (1):                                                                 #
#   (2):                                                                 #
#                                                                        #
#========================================================================*/
def frac_e(log10Ms):
    A = 0.46
    gam_l = 3.75
    logMchar_l = 11.09
    x0l = 0.09
    gam_r = 1.51
    logMchar_r = 10.38
    x0r = 0.462

    xl = log10Ms - logMchar_l
    xr = log10Ms - logMchar_r

    Sig_l = (1 - A) * (np.exp(gam_l * (xl + x0l)) / (1 + np.exp(gam_l * (xl + x0l))))
    Sig_r = A * (np.exp(gam_r * (xr - x0r)) / (1 + np.exp(gam_r * (xr - x0r))))

    Sigmoid = Sig_l + Sig_r
    return Sigmoid


#################################################################################

# DISTRIBUTION INTEGRALS*/
#=======================================================================!
# FUNCTION:  var_diff                                                    !
#   This function makes only makes the difference among two variables.   !
#                                                                        !
#=======================================================================*/
def var_diff( val1,  val2):
  return val1 - val2


#=======================================================================!
# FUNCTION:  Pcumu_Schech_logR_LTGs                                      !
#   This function computes the integral of the logRHI and logRH2         !
#   distributions (LTGs case).                                           !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
def Pcumu_Schech_logR_LTGs( logMs,  params,  logR1,  logR2,  flag_Hj):
    N=110
    s = (N+1)*[0]
    integral=0.0
    ntegral1=0.0
    integral2=0.0
    delta_logR = (logR2 - logR1) / N

    # Integrating function */
    for i in range(N+1):
      logR_dummy = logR1 + (delta_logR * i)
      s[i] = P_MHj_Ms_blue_LTGs(logR_dummy + logMs, logMs, params, flag_Hj)

      if (i >= 1 and i <= N-1):
          integral1+=s[i]

      if( np.sqrt(np.powf(np.log10(integral1) - np.log10(integral2),2)) < 1E-5):
          break

      integral2=integral1

    s[N] = P_MHj_Ms_blue_LTGs(logR2 + logMs, logMs, params, flag_Hj)
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])
    return integral

#=======================================================================!
# FUNCTION:  Pcumu_Schech_logR_ETGs                                      !
#   This function computes the integral of the logRHI and logRH2         !
#   distributions (LTGs case).                                           !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
def Pcumu_Schech_logR_ETGs( logMs,  params,  logR1,  logR2,  flag_Hj):
    N=110
    s = (N+1)*[0]
    integral=0.0
    integral1=0.0
    integral2=0.0

    delta_logR = (logR2 - logR1) / N

    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        s[i] = P_MHj_Ms_red_ETGs(logR_dummy + logMs, logMs, params, flag_Hj)
        if(i>=1 and i<=N-1):
            integral1+=s[i]
        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2))<1E-5):
            break

        integral2=integral1

    s[N] = P_MHj_Ms_red_ETGs(logR2 + logMs, logMs, params, flag_Hj)
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
# FUNCTION:  FirstMom_Schech_logR_LTGs                                   !
#   This function computes the First Moment of the logRHI and logRH2     !
#   Schechter distributions (LTGs case).                                 !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
def FirstMom_P_RHj_LTGs( logMs,  params,  logR1,  logR2,  flag_Hj):
    N=510
    s = (N+1)*[0]
    integral=0.0
    integral1=0.0
    integral2=0.0

    delta_logR = (logR2 - logR1) / N

    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        s[i] = P_MHj_Ms_blue_LTGs(logR_dummy + logMs, logMs, params, flag_Hj) * logR_dummy

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2)) < 1E-5):
            break

        integral2=integral1

    s[N] = P_MHj_Ms_blue_LTGs(logR2 + logMs, logMs, params, flag_Hj) * logR2
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
# FUNCTION:  SecondMom_P_RHj_LTGs                                        !
#   This function computes the Second Moment of the logRHI and logRH2    !
#   distributions (LTGs case).                                           !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
def SecondMom_P_RHj_LTGs( logMs,  params,  logR1,  logR2,  flag_Hj):
    N=510
    s = (N+1)*[0]
    integral=0.0
    integral1=0.0
    integral2=0.0

    delta_logR = (logR2 - logR1) / N

    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)

        Term_prod1 = logR_dummy - FirstMom_P_RHj_LTGs(logMs, params, logR1, logR2, flag_Hj)
        Term_prod1 = Term_prod1 * Term_prod1
        s[i] = P_MHj_Ms_blue_LTGs(logR_dummy + logMs, logMs, params, flag_Hj) * Term_prod1

        if(i>=1 and i<=N-1):
            integral1+=s[i]
        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2)) < 1E-5):
            break

        integral2=integral1

    Term_prod1 = logR2 - FirstMom_P_RHj_LTGs(logMs, params, logR1, logR2, flag_Hj)
    Term_prod1 = Term_prod1 * Term_prod1
    s[N] = P_MHj_Ms_blue_LTGs(logR2 + logMs, logMs, params, flag_Hj) * Term_prod1
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
# FUNCTION:  FirstMom_Schech_logR_ETGs                                   !
#   This function computes the First Moment of the logRHI and logRH2     !
#   Schechter distributions (LTGs case).                                 !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
def FirstMom_P_RHj_ETGs( logMs,  params,  logR1,  logR2,  flag_Hj):
    N=1010
    s = (N+1)*[0]
    integral=0.0
    integral1=0.0
    integral2=0.0

    delta_logR = (logR2 - logR1) / N

    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        s[i] = P_MHj_Ms_red_ETGs(logR_dummy + logMs, logMs, params, flag_Hj) * logR_dummy

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2))<1E-5):
            integral2=integral1

    s[N] = P_MHj_Ms_red_ETGs(logR2 + logMs, logMs, params, flag_Hj) * logR2
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
# FUNCTION:  SecondMom_P_RHj_ETGs                                        !
#   This function computes the Second Moment of the logRHI and logRH2    !
#   distributions (LTGs case).                                           !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
def SecondMom_P_RHj_ETGs( logMs,  params,  logR1,  logR2,  flag_Hj):
    N=1010
    s = (N+1)*[0]
    integral=0.0
    integral1=0.0
    integral2=0.0

    delta_logR = (logR2 - logR1) / N

    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        Term_prod1 = logR_dummy - FirstMom_P_RHj_ETGs(logMs, params, logR1, logR2, flag_Hj)
        Term_prod1 = Term_prod1 * Term_prod1
        s[i] = P_MHj_Ms_red_ETGs(logR_dummy + logMs, logMs, params, flag_Hj) * Term_prod1

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2))<1E-5):
            break

        integral2=integral1


    Term_prod1 = logR2 - FirstMom_P_RHj_ETGs(logMs, params, logR1, logR2, flag_Hj)
    Term_prod1 = Term_prod1 * Term_prod1
    s[N] = P_MHj_Ms_red_ETGs(logR2 + logMs, logMs, params, flag_Hj) * Term_prod1
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral


#=======================================================================!
# FUNCTION:  FirstMom_P_RHj_all                                         !
#   This function computes the First Moment of the logRHI and logRH2     !
#   Schechter distributions (LTGs case).                                 !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
def FirstMom_P_RHj_all( logMs,  params_LT,  params_ET,  logR1,  logR2,  flag_Hj):
    N=510
    s = (N+1)*[0]
    integral=0.0
    integral1=0.0
    integral2=0.0

    delta_logR = (logR2 - logR1) / N

    #Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        s[i] = P_MHj_Ms_all(logR_dummy + logMs, logMs, params_LT, params_ET, flag_Hj) * logR_dummy

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2))<1E-5):
            break

        integral2=integral1

    s[N] = P_MHj_Ms_all(logR2 + logMs, logMs, params_LT, params_ET, flag_Hj) * logR2
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
# FUNCTION:  SecondMom_P_RHj_ETGs                                        !
#   This function computes the Second Moment of the logRHI and logRH2    !
#   distributions (LTGs case).                                           !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
def SecondMom_P_RHj_all( logMs,  params_LT,  params_ET,  logR1,  logR2,  flag_Hj):
    N=510
    s = (N+1)*[0]
    integral=0.0
    integral1=0.0
    integral2=0.0

    delta_logR = (logR2 - logR1) / N

    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        Term_prod1 = logR_dummy - FirstMom_P_RHj_all(logMs, params_LT, params_ET, logR1, logR2, flag_Hj)
        Term_prod1 = Term_prod1 * Term_prod1
        s[i] = P_MHj_Ms_all(logR_dummy + logMs, logMs, params_LT, params_ET, flag_Hj) * Term_prod1

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2))<1E-5):
            break

        integral2=integral1

    Term_prod1 = logR2 - FirstMom_P_RHj_all(logMs, params_LT, params_ET, logR1, logR2, flag_Hj)
    Term_prod1 = Term_prod1 * Term_prod1
    s[N] = P_MHj_Ms_all(logR2 + logMs, logMs, params_LT, params_ET, flag_Hj) * Term_prod1
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
# FUNCTION:  percentiles_LTGs                                            !
#   Computes percentiles of cold gas conditional distributions.          !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
def percentiles_LTGs( logMs,  params,  logR1,  logR2,  flag_Hj, p1, p2, p3):
    x1=0
    Nbins=40
    Delta = (logR2 - logR1) / Nbins

    for j in range (1,4):
        if(j == 1):
            Prandom = p1
        elif (j == 2):
            Prandom = p2
        elif (j == 3):
            Prandom = p3

        for i in range (Nbins + 1):
            logRFalse = (logR1) + Delta*i
            Pcumu = Pcumu_Schech_logR_LTGs(logMs, params, logRFalse, logR2, flag_Hj)
            f = var_diff(Prandom, Pcumu)

            if(f>=0):
                x2 = logRFalse
                break

            x1 = logRFalse

        x3 = 1.0

        for i in range(201):
            Xm = 0.5 * (x1 + x2)
            if(np.sqrt(np.power(Xm - x3, 2))<1E-5):
                break

            Pcumu = Pcumu_Schech_logR_LTGs(logMs, params, Xm, logR2, flag_Hj)
            Pcumu2 = Pcumu_Schech_logR_LTGs(logMs, params, x1, logR2, flag_Hj)

            if (var_diff(Prandom, Pcumu)*var_diff(Prandom, Pcumu2)<=0):
                x2 = Xm
            else:
                x1 = Xm

            x3 = Xm

        if(j == 1):
            per_left = Xm
        elif (j == 2):
            per_mid = Xm
        elif (j == 3):
            per_right = Xm

    return per_left, per_mid, per_right

#=======================================================================!
# FUNCTION:  percentiles_ETGs                                            !
#   Computes percentiles of cold gas conditional distributions.          !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
def percentiles_ETGs( logMs,  params,  logR1,  logR2,  flag_Hj, p1, p2, p3):
    x1=0
    Nbins=40
    Delta = (logR2 - logR1) / Nbins

    for j in range (1,4):
        if(j == 1):
            Prandom = p1
        elif (j == 2):
            Prandom = p2
        elif (j == 3):
            Prandom = p3

        for i in range(Nbins + 1):
            logRFalse = (logR1) + Delta*i
            Pcumu = Pcumu_Schech_logR_ETGs(logMs, params, logRFalse, logR2, flag_Hj)
            f = var_diff(Prandom, Pcumu)

            if (f >= 0):
                x2 = logRFalse
                break

        x1 = logRFalse

        x3 = 1.0

        for i in range(201):
            Xm = 0.5 * (x1 + x2)
            if(np.sqrt(np.power(Xm - x3, 2)) < 1E-5):
                break

            Pcumu = Pcumu_Schech_logR_ETGs(logMs, params, Xm, logR2, flag_Hj)
            Pcumu2 = Pcumu_Schech_logR_ETGs(logMs, params, x1, logR2, flag_Hj)

            if(var_diff(Prandom, Pcumu) * var_diff(Prandom, Pcumu2) <= 0):
                x2 = Xm
            else:
                x1 = Xm

            x3 = Xm

        if(j == 1):
            per_left = Xm
        elif (j == 2):
            per_mid = Xm
        elif (j == 3):
            per_right = Xm

    return per_left, per_mid, per_right
