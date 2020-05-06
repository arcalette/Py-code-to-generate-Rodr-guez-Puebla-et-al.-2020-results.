# -*- encoding: utf-8 -*-
"""
Module: module_gas.py
Author: Ángel Rubén Calette Morín.

Description:
This function is the upper limit in the integral of the TH function,
or the lower limit of Schechter function.                           

"""

import numpy as np
from scipy.special import gamma, gammaincc

#/=======================================================================!
# FUNCTION:  GSMF_RP19                                                   #
#   This is the fit of the Galaxy Stellar Mass Function constrained      #
#   in Rodriguez-Puebla et al.                                           #
#                                                                        #
#                                                                        #
#  Notes:                                                                #
#   (1):                                                                 #
#                                                                        #
#=======================================================================*/
cdef float dpl(float logphi_st, float alpha, float gammma, float delta, float logM_st, float log10Ms):
    cdef float logphi
    logphi = logphi_st + (alpha + 1) * ( log10Ms - logM_st ) - ( delta - alpha ) * np.log10(1. + np.power(10, gammma * (log10Ms-logM_st) )) / gammma - np.log10(np.log10(2.71828182846))
      
    return np.power(10,logphi)

cdef float Schech(float logphi_st, float alpha, float beta, float logM_st, float log10Ms):
    cdef float logphi
  
    logphi=logphi_st+(alpha+1)*(log10Ms-logM_st)-np.power(10,beta*(log10Ms-logM_st)) * np.log10(np.exp(1))-np.log10(np.log10(2.71828182846))
  
    return np.power(10,logphi)
  

cdef float GSMF_RP19(x, float log10Ms):
    cdef float logphi
    logphi = Schech(x[1],x[2],x[3],x[4],log10Ms) + dpl(x[5],x[6],x[7],x[8],x[9],log10Ms) # Schechter + DPL
    return np.power(10, logphi) 

#/=======================================================================!
def GSMF(x, log10Ms):
    return GSMF_RP19(x, log10Ms)

#/=======================================================================!
# FUNCTION:  x2_P_TH_dlogR                                              #
#   This function is the upper limit in the integral of the TH function,#
#   or the lower limit of Schechter function.                           #
#                                                                       #
#  Notes:                                                               #
#   (1):                                                                #
#                                                                       #
#=======================================================================*/
cdef float x2_P_TH_dlogR(float logMs, float slope, float b):
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
cdef float F_P_TH_dlogR(float logMs, float slope, float b):
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
cdef float logRstr(float logMs, float C, float logMbr, float a, float b):
    cdef float ratio, Term1, Term2
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
cdef float alpha_logMs(float logMs, float slope, float b):
    return (slope*logMs) + b

#/=======================================================================!
# FUNCTION:  phistr                                                     #
#   This function is a stellar mass dependent  functional form          #
#   of the phiStr parameter for Schechter function (HI and H2) for LTGs #
#                                                                       #
#  Notes:                                                               #
#   (1): Remember in this case phiStr = 1 / Gamma(1 + alpha)            #
#   (2): alpha = Schechter slope parameter.                             #
#   (3): phiStr = Built in such way that integral of Schechter is 1.    #
#                                                                       #
#=======================================================================*/
cdef float phistr(float alpha):
    return 1.0 / gamma(1.0 + alpha)


# DISTRIBUTIONS AND DISTRIBUTIONS INTEGRALS #
#/=======================================================================!
# FUNCTION:  Sx_logx                                                    #
#                                                                       #
#  Notes:                                                               #
#   (1):                                                                #
#                                                                       #
#=======================================================================*/
cdef float Sx_logx(float logRHj, float alpha, float logRHj_char):
    cdef float ratio
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
cdef float Int_Sx_logx(float alpha, float logRst, float logRHj_min, float logRHj_max):
    cdef int N=50
    cdef float integral=0.0, integral1=0.0
    cdef float logRmin, logRmax, delta_logRHj 
    cdef float logRHj_x, PDF_MHj
    s = (N+1)*[0.0]
    
    logRmin = logRHj_min
    logRmax = logRHj_max
    delta_logRHj = (logRmax - logRmin) / (N * 1.0)
    
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
cdef float Schech_RHj_lin(float logRHj, float alpha, float logRHj_char):
    cdef float ratio, phist
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
def Schech_RHj_log(float logRHj, float alpha, float logRHj_char):
#cdef float Schech_RHj_log(float logRHj, float alpha, float logRHj_char):
    cdef float ratio, phist
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
cdef float P_MHj_Ms_blue_LTGs(float logMHj, float logMs, params, float flag_Hj):
    cdef float logRHj, logMs_x, alpha, logRtr, Int_den, PDF_RHj
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

def P_MHj_Ms_late(float logMHj, float logMs, params, float flag_Hj):
    return P_MHj_Ms_blue_LTGs(logMHj, logMs, params, flag_Hj)

#=======================================================================*/
cdef float Int_GSMF_morph(float logMs_min, float logMs_max, x, float morph_case): #1=late, 2=early
    cdef int N=110
    cdef float logMs_dummy,delta_logMs
    cdef float integral=0.0, integral1=0.0
    cdef float phi_dummy
    
    s = (N+1)*[0.0]
    delta_logMs = (logMs_max - logMs_min) / N
    
    # Integrating function */
    for i in range(N + 1):
        logMs_dummy = logMs_min + (delta_logMs * i)
        phi_dummy = GSMF_RP19(x, logMs_dummy) * ( 1.0 - frac_e(logMs_dummy) ) if morph_case == 1.0  else  GSMF_RP19(x, logMs_dummy) * frac_e(logMs_dummy) 
        
        s[i] = phi_dummy
        if(i>=1 and i<=N-1):
            integral1+=s[i]
    
    integral =  0.5 * delta_logMs * (s[0] + (2 * integral1) + s[N])
    return integral


#=======================================================================!
def Int_norm_Mtype(float logMs_min, float logMs_max, x, float morph_case):
    return Int_GSMF_morph(logMs_min, logMs_max, x, morph_case)

#=======================================================================*/
cdef float Int_GSMF(float logMs_min, float logMs_max, x): #1=late, 2=early
    cdef int N=110
    cdef float logMs_dummy,delta_logMs
    cdef float integral=0.0, integral1=0.0
    cdef float phi_dummy
    
    s = (N+1)*[0.0]
    delta_logMs = (logMs_max - logMs_min) / N
    
    # Integrating function */
    for i in range(N + 1):
        logMs_dummy = logMs_min + (delta_logMs * i)
        phi_dummy = GSMF_RP19(x, logMs_dummy)  
        
        s[i] = phi_dummy
        if(i>=1 and i<=N-1):
            integral1+=s[i]
    
    integral =  0.5 * delta_logMs * (s[0] + (2 * integral1) + s[N])
    return integral

#=======================================================================!
def Int_norm(float logMs_min, float logMs_max, x):
    return Int_GSMF(logMs_min, logMs_max, x)

#=======================================================================!
# FUNCTION: ave_P_MHj_Ms_blue_LTGs_c                                    #
#                                                                       #
#  Notes:                                                               #
#   (1):                                                                #
#                                                                       #
#=======================================================================*/
cdef float ave_P_MHj_Ms_all_c(float logMHj, params_LT, params_ET, x, float flag_Hj, float logM_ini, float logM_end, int N, float norm):
    cdef float integral=0, integral1=0, dlogM, logM_x, phi_dummy
    s = (N+1)*[0.0]
    #norm = Int_GSMF_morph(logM_ini, logM_end, x, 1.0);
    dlogM = (logM_end - logM_ini) / N
    
    for i in range(N + 1):
        logM_x = logM_ini + dlogM * i
        phi_dummy =  GSMF_RP19(x, logM_x)
        PDF_MHj = P_MHj_Ms_all(logMHj, logM_x, params_LT, params_ET, flag_Hj)
        s[i] =  (PDF_MHj *  phi_dummy) / norm
        if(i>0 and i<N):
            integral1+=s[i]

    
    integral = 0.5 * dlogM * ( s[0] + 2. * integral1 + s[N] )
    
    return integral 
    
#=======================================================================!
def ave_P_MHj_Ms_all(float logMHj, params_LT, params_ET, x, float flag_Hj, float logM_ini, float logM_end, int N, float norm):
    return ave_P_MHj_Ms_all_c(logMHj, params_LT, params_ET, x, flag_Hj, logM_ini, logM_end, N, norm)

#=======================================================================!
# FUNCTION: ave_P_MHj_Ms_blue_LTGs_c                                    #
#                                                                       #
#  Notes:                                                               #
#   (1):                                                                #
#                                                                       #
#=======================================================================*/
cdef float ave_P_MHj_Ms_blue_LTGs_c(float logMHj, params, x, float flag_Hj, float logM_ini, float logM_end, int N, float norm):
    cdef float integral=0, integral1=0, dlogM, logM_x, phi_dummy
    s = (N+1)*[0.0]
    #norm = Int_GSMF_morph(logM_ini, logM_end, x, 1.0);
    dlogM = (logM_end - logM_ini) / N
    
    for i in range(N + 1):
        logM_x = logM_ini + dlogM * i
        phi_dummy =  GSMF_RP19(x, logM_x) * ( 1.0 - frac_e(logM_x) )
        PDF_MHj = P_MHj_Ms_blue_LTGs(logMHj, logM_x, params, flag_Hj)
        s[i] =  (PDF_MHj *  phi_dummy) / norm
        if(i>0 and i<N):
            integral1+=s[i]

    
    integral = 0.5 * dlogM * ( s[0] + 2. * integral1 + s[N] )
    
    return integral 
    
#=======================================================================!
def ave_P_MHj_Ms_blue_LTGs(float logMHj, params, x, float flag_Hj, float logM_ini, float logM_end, int N, float norm):
    return ave_P_MHj_Ms_blue_LTGs_c(logMHj, params, x, flag_Hj, logM_ini, logM_end, N, norm)

#=======================================================================!
# FUNCTION: ave_P_MHj_Ms_red_ETGs_c                                    #
#                                                                       #
#  Notes:                                                               #
#   (1):                                                                #
#                                                                       #
#=======================================================================*/
cdef float ave_P_MHj_Ms_red_ETGs_c(float logMHj, params, x, float flag_Hj, float logM_ini, float logM_end, int N, float norm):
    cdef float integral=0, integral1=0, dlogM, logM_x, phi_dummy
    s = (N+1)*[0.0]
    #norm = Int_GSMF_morph(logM_ini, logM_end, x, 1.0);
    dlogM = (logM_end - logM_ini) / N
    
    for i in range(N + 1):
        logM_x = logM_ini + dlogM * i
        phi_dummy =  GSMF_RP19(x, logM_x) *  frac_e(logM_x) 
        PDF_MHj = P_MHj_Ms_red_ETGs(logMHj, logM_x, params, flag_Hj)
        s[i] =  (PDF_MHj *  phi_dummy) / norm
        if(i>0 and i<N):
            integral1+=s[i]

    
    integral = 0.5 * dlogM * ( s[0] + 2. * integral1 + s[N] )
    
    return integral 
    
#=======================================================================!
def ave_P_MHj_Ms_red_ETGs(float logMHj, params, x, float flag_Hj, float logM_ini, float logM_end, int N, float norm):
    return ave_P_MHj_Ms_red_ETGs_c(logMHj, params, x, flag_Hj, logM_ini, logM_end, N, norm)

#=======================================================================!
# FUNCTION:  P_MHj_Ms_red_ETGs                                          #
#                                                                       #
#  Notes:                                                               #
#   (1):                                                                #
#                                                                       #
#=======================================================================*/
cdef float P_MHj_Ms_red_ETGs(float logMHj, float logMs, params, float flag_Hj):
    cdef float width_P_TH=1.0, logRHj, logMs_x, PDF_RHj_TH, x2_P_TH, x1_P_TH, alpha
    cdef float logRtr, x, Gamma, Q, err, Int_den, PDF_RHj
    logRHj = logMHj - logMs

    if (flag_Hj == 1.0): #HI
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
    Q, err = gammq_err_call(1.0 + alpha, x)
    Int_den = Q * Gamma if err == 0.0 else  Int_Sx_logx(alpha, logRtr, x2_P_TH, 4.0)

    # Use Schechter or TopHat? */
    PDF_RHj = Sx_logx(logRHj, alpha, logRtr) * ( (1.0 - PDF_RHj_TH) / Int_den) if Int_den >= 0 else 0
    PDF_RHj = PDF_RHj_TH if logRHj <= x2_P_TH and logRHj >= x1_P_TH else PDF_RHj
    PDF_RHj = 0 if logRHj <= x1_P_TH else PDF_RHj
    return PDF_RHj
    
#=======================================================================!
def P_MHj_Ms_red_early(float logMHj, float logMs, params, float flag_Hj):
    return P_MHj_Ms_red_ETGs(logMHj, logMs, params, flag_Hj)

#=======================================================================!
# FUNCTION:  P_MHj_Ms_all                                               #
#                                                                       #
#  Notes:                                                               #
#   (1):                                                                #
#                                                                       #
#=======================================================================*/
cdef float P_MHj_Ms_all(float logMHj, float logMs, params_LT, params_ET, float flag_Hj):
    cdef float logRHj, PDF_RHj_LT, PDF_RHj_ET, PDF_RHj
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
cdef float gammq_err(float a, float x):
    cdef float err, Q
    err = 0.0
    Q = 0.0

    if (x < 0.0 or a <= 0.0):
        err = 1.0
    else:
        Q = gammaincc(a,x)

    return Q

#=======================================================================!
# FUNCTION:  gammq_err2                                                 !
#                                                                       !
#                                                                       !
#                                                                       !
#  Notes:                                                               !
#  (1):                                                                 !
#=======================================================================*/
cdef float gammq_err2(float a, float x):
    cdef float err, Q
    err = 0.0
    Q = 0.0

    if (x < 0.0 or a <= 0.0):
        err = 1.0
    else:
        Q = gammaincc(a,x)

    return err

#=======================================================================!
def gammq_err_call(float a, float x):
    cdef float Q, err
    Q = gammq_err( a, x )
    err = gammq_err2( a, x )
    return Q, err

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
cdef float frac_e(float log10Ms):
    cdef float A, gam_l, logMchar_l, x0l, gam_r, logMchar_r, x0r, xl, xr, Sig_l, Sig_r
    cdef float Sigmoid
    
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
def frac_ETGs(logMs):
    return frac_e(logMs)

#################################################################################

# DISTRIBUTION INTEGRALS*/
#=======================================================================!
# FUNCTION:  var_diff                                                    !
#   This function makes only makes the difference among two variables.   !
#                                                                        !
#=======================================================================*/
def var_diff( float val1,  float val2 ):
#cdef float var_diff( float val1,  float val2):
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
def Pcumu_Schech_logR_LTGs(  float logMs, params, float logR1, float logR2, float flag_Hj):
#cdef float Pcumu_Schech_logR_LTGs( float logMs, params, float logR1, float logR2, float flag_Hj):
    cdef int N=110
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy
    s = (N+1)*[0]
    
    delta_logR = (logR2 - logR1) / N

    # Integrating function */
    for i in range(N+1):
      logR_dummy = logR1 + (delta_logR * i)
      s[i] = P_MHj_Ms_blue_LTGs(logR_dummy + logMs, logMs, params, flag_Hj)

      if (i >= 1 and i <= N-1):
          integral1+=s[i]

      if( np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2)) < 1E-5):
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
def Pcumu_Schech_logR_ETGs( float logMs,  params,  float logR1,  float logR2,  float flag_Hj ):
#cdef float Pcumu_Schech_logR_ETGs( float logMs,  params,  float logR1,  float logR2,  float flag_Hj):
    cdef int i, N=110
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy
    
    s = (N+1)*[0]

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
# FUNCTION:  Pcumu_Schech_logR_all                                      !
#   This function computes the integral of the logRHI and logRH2        !
#   distributions (all galaxies case).                                  !
#                                                                       !
#                                                                       !
#                                                                       !
#  Notes:                                                               !
#                                                                       !
#                                                                       !
#=======================================================================*/
#def Pcumu_Schech_logR_all(  float logMs, params_LTGs, params_ETGs, float logR1, float logR2, float flag_Hj):
cdef float Pcumu_Schech_logR_all_c( float logMs, params_LTGs, params_ETGs, float logR1, float logR2, float flag_Hj):
    cdef int N=110
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy
    s = (N+1)*[0]
    
    delta_logR = (logR2 - logR1) / N

    # Integrating function */
    for i in range(N+1):
      logR_dummy = logR1 + (delta_logR * i)
      s[i] = P_MHj_Ms_all(logR_dummy + logMs, logMs, params_LTGs, params_ETGs, flag_Hj)

      if (i >= 1 and i <= N-1):
          integral1+=s[i]

      if( np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2)) < 1E-5):
          break

      integral2=integral1

    s[N] = P_MHj_Ms_all(logR2 + logMs, logMs, params_LTGs, params_ETGs, flag_Hj)
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
cdef float FirstMom_P_logRHj_LTGs( float logMs,  params,  float logR1,  float logR2,  float flag_Hj):
    cdef int N=510
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy
    s = (N+1)*[0.0]

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
def FM_P_logRHj_LTGs( float logMs,  params,  float logR1,  float logR2,  float flag_Hj ):
    return FirstMom_P_logRHj_LTGs( logMs,  params,  logR1,  logR2,  flag_Hj)

#=======================================================================!
# FUNCTION:  SecondMom_P_logRHj_LTGs                                        !
#   This function computes the Second Moment of the logRHI and logRH2    !
#   distributions (LTGs case).                                           !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
cdef float SecondMom_P_logRHj_LTGs( float logMs, params,  float logR1,  float logR2,  float flag_Hj):
    cdef int N=510
    cdef integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy, logR_ave
    cdef float Term_prod1 
    s = (N+1)*[0]

    delta_logR = (logR2 - logR1) / (N * 1.0)
    logR_ave = FirstMom_P_logRHj_LTGs(logMs, params, logR1, logR2, flag_Hj)
    
    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)

        Term_prod1 = logR_dummy - logR_ave
        Term_prod1 = Term_prod1 * Term_prod1
        s[i] = P_MHj_Ms_blue_LTGs(logR_dummy + logMs, logMs, params, flag_Hj) * Term_prod1

        if(i>=1 and i<=N-1):
            integral1+=s[i]
        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2)) < 1E-5):
            break

        integral2=integral1

    Term_prod1 = logR2 - logR_ave
    Term_prod1 = Term_prod1 * Term_prod1
    s[N] = P_MHj_Ms_blue_LTGs(logR2 + logMs, logMs, params, flag_Hj) * Term_prod1
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
def SM_P_logRHj_LTGs( float logMs,  params,  float logR1,  float logR2,  float flag_Hj ):
    return SecondMom_P_logRHj_LTGs(  logMs, params,  logR1,  logR2,  flag_Hj )

#=======================================================================!
# FUNCTION:  FirstMom_Schech_logMHj_LTGs                                  !
#   This function computes the First Moment of the logRHI and logRH2     !
#   Schechter distributions (LTGs case).                                 !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
cdef float FirstMom_Schech_logMHj_LTGs( float logMs,  params,  float logR1,  float logR2,  float flag_Hj):
    cdef int N=510
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy, logMHj_dummy
    s = (N+1)*[0.0]

    delta_logR = (logR2 - logR1) / N

    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        logMHj_dummy = logR_dummy + logMs
        s[i] = P_MHj_Ms_blue_LTGs(logR_dummy + logMs, logMs, params, flag_Hj) * logMHj_dummy

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2)) < 1E-5):
            break

        integral2=integral1
    
    logMHj_dummy = logR2 + logMs
    s[N] = P_MHj_Ms_blue_LTGs(logR2 + logMs, logMs, params, flag_Hj) * logMHj_dummy
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
def FM_P_logMHj_LTGs( float logMs,  params,  float logR1,  float logR2,  float flag_Hj ):
    return FirstMom_Schech_logMHj_LTGs( logMs,  params,  logR1,  logR2,  flag_Hj)

#=======================================================================!
# FUNCTION:  SecondMom_P_logMHj_LTGs                                     !
#   This function computes the Second Moment of the logRHI and logRH2    !
#   distributions (LTGs case).                                           !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
cdef float SecondMom_P_logMHj_LTGs( float logMs, params,  float logR1,  float logR2,  float flag_Hj):
    cdef int N=510
    cdef integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy
    cdef float Term_prod1, logMHj_dummy, logM_ave 
    s = (N+1)*[0]

    delta_logR = (logR2 - logR1) / (N * 1.0)
    logM_ave = FirstMom_Schech_logMHj_LTGs(logMs, params, logR1, logR2, flag_Hj)
    
    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        logMHj_dummy = logR_dummy + logMs

        Term_prod1 = logMHj_dummy - logM_ave
        Term_prod1 = Term_prod1 * Term_prod1
        s[i] = P_MHj_Ms_blue_LTGs(logR_dummy + logMs, logMs, params, flag_Hj) * Term_prod1

        if(i>=1 and i<=N-1):
            integral1+=s[i]
        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2)) < 1E-5):
            break

        integral2=integral1

    logMHj_dummy = logR2 + logMs
    Term_prod1 = logMHj_dummy - logM_ave
    Term_prod1 = Term_prod1 * Term_prod1
    s[N] = P_MHj_Ms_blue_LTGs(logR2 + logMs, logMs, params, flag_Hj) * Term_prod1
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
def SM_P_logMHj_LTGs( float logMs,  params,  float logR1,  float logR2,  float flag_Hj ):
    return SecondMom_P_logMHj_LTGs(  logMs, params,  logR1,  logR2,  flag_Hj )

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
cdef float FirstMom_P_logRHj_ETGs( float logMs, params, float logR1, float logR2, float flag_Hj):
    cdef int N=1510
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy
    s = (N+1)*[0.0]

    delta_logR = (logR2 - logR1) / (N * 1.0)

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
def FM_P_logRHj_ETGs( float logMs, params, float logR1, float logR2, float flag_Hj ):
    return FirstMom_P_logRHj_ETGs( logMs, params, logR1, logR2, flag_Hj )

#=======================================================================!
# FUNCTION:  SecondMom_P_logRHj_ETGs                                        !
#   This function computes the Second Moment of the logRHI and logRH2    !
#   distributions (LTGs case).                                           !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
cdef float SecondMom_P_logRHj_ETGs( float logMs, params, float logR1, float logR2, float flag_Hj):
    cdef int N=1510
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy, Term_prod1, logR_ave
    s = (N+1)*[0.0]

    delta_logR = (logR2 - logR1) / N
    logR_ave = FirstMom_P_logRHj_ETGs(logMs, params, logR1, logR2, flag_Hj)
    
    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        Term_prod1 = logR_dummy - logR_ave
        Term_prod1 = Term_prod1 * Term_prod1
        s[i] = P_MHj_Ms_red_ETGs(logR_dummy + logMs, logMs, params, flag_Hj) * Term_prod1

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2))<1E-5):
            break

        integral2=integral1


    Term_prod1 = logR2 - logR_ave
    Term_prod1 = Term_prod1 * Term_prod1
    s[N] = P_MHj_Ms_red_ETGs(logR2 + logMs, logMs, params, flag_Hj) * Term_prod1
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
def SM_P_logRHj_ETGs( float logMs, params, float logR1, float logR2, float flag_Hj ):
    return SecondMom_P_logRHj_ETGs( logMs, params, logR1, logR2, flag_Hj )

#=======================================================================!
# FUNCTION:  FirstMom_P_logMHj_ETGs                                      !
#   This function computes the First Moment of the logRHI and logRH2     !
#   Schechter distributions (LTGs case).                                 !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
cdef float FirstMom_P_logMHj_ETGs( float logMs, params, float logR1, float logR2, float flag_Hj):
    cdef int N=510
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy, logMHj_dummy
    s = (N+1)*[0.0]

    delta_logR = (logR2 - logR1) / (N * 1.0)

    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        logMHj_dummy = logR_dummy + logMs
        s[i] = P_MHj_Ms_red_ETGs(logR_dummy + logMs, logMs, params, flag_Hj) * logMHj_dummy

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2))<1E-5):
            integral2=integral1

    logMHj_dummy = logR2 + logMs
    s[N] = P_MHj_Ms_red_ETGs(logR2 + logMs, logMs, params, flag_Hj) * logMHj_dummy
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
def FM_P_logMHj_ETGs( float logMs, params, float logR1, float logR2, float flag_Hj ):
    return FirstMom_P_logMHj_ETGs( logMs, params, logR1, logR2, flag_Hj )

#=======================================================================!
# FUNCTION:  SecondMom_P_logMHj_ETGs                                        !
#   This function computes the Second Moment of the logRHI and logRH2    !
#   distributions (LTGs case).                                           !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
cdef float SecondMom_P_logMHj_ETGs( float logMs, params, float logR1, float logR2, float flag_Hj):
    cdef int N=510
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy, Term_prod1, logMHj_dummy, logMHj_ave, logR_ave
    s = (N+1)*[0.0]

    delta_logR = (logR2 - logR1) / N
    logMHj_ave = FirstMom_P_logMHj_ETGs(logMs, params, logR1, logR2, flag_Hj)
    
    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        logMHj_dummy = logR_dummy + logMs
        Term_prod1 = logMHj_dummy - logMHj_ave
        Term_prod1 = Term_prod1 * Term_prod1
        s[i] = P_MHj_Ms_red_ETGs(logR_dummy + logMs, logMs, params, flag_Hj) * Term_prod1

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2))<1E-5):
            break

        integral2=integral1

    logMHj_dummy = logR2 + logMs
    Term_prod1 = logMHj_dummy - logMHj_ave
    Term_prod1 = Term_prod1 * Term_prod1
    s[N] = P_MHj_Ms_red_ETGs(logR2 + logMs, logMs, params, flag_Hj) * Term_prod1
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
def SM_P_logMHj_ETGs( float logMs, params, float logR1, float logR2, float flag_Hj ):
    return SecondMom_P_logMHj_ETGs( logMs, params, logR1, logR2, flag_Hj )

#=======================================================================!
# FUNCTION:  FirstMom_P_logRHj_all                                         !
#   This function computes the First Moment of the logRHI and logRH2     !
#   Schechter distributions (LTGs case).                                 !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
def FirstMom_P_logRHj_all(  float logMs,  params_LT,  params_ET, float logR1, float logR2, float flag_Hj):
#cdef float FirstMom_P_logRHj_all( float logMs,  params_LT,  params_ET, float logR1, float logR2, float flag_Hj):
    cdef int N=510
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy
    s = (N+1)*[0]

    delta_logR = (logR2 - logR1) / (N * 1.0)

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
# FUNCTION:  SecondMom_P_logRHj_ETGs                                        !
#   This function computes the Second Moment of the logRHI and logRH2    !
#   distributions (LTGs case).                                           !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
def SecondMom_P_logRHj_all( logMs,  params_LT,  params_ET,  logR1,  logR2,  flag_Hj):
#cdef float SecondMom_P_logRHj_all( float logMs,  params_LT,  params_ET,  float logR1,  float logR2,  float flag_Hj):
    cdef int N=510
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy, Term_prod1, logR_ave
    s = (N+1)*[0.0]

    delta_logR = (logR2 - logR1) / N
    logR_ave = FirstMom_P_logRHj_all(logMs, params_LT, params_ET, logR1, logR2, flag_Hj)
    
    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        Term_prod1 = logR_dummy - logR_ave
        Term_prod1 = Term_prod1 * Term_prod1
        s[i] = P_MHj_Ms_all(logR_dummy + logMs, logMs, params_LT, params_ET, flag_Hj) * Term_prod1

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2))<1E-5):
            break

        integral2=integral1

    Term_prod1 = logR2 - logR_ave
    Term_prod1 = Term_prod1 * Term_prod1
    s[N] = P_MHj_Ms_all(logR2 + logMs, logMs, params_LT, params_ET, flag_Hj) * Term_prod1
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
# FUNCTION:  FirstMom_P_logMHj_all_c                                     !
#   This function computes the First Moment of the logRHI and logRH2     !
#   Schechter distributions (LTGs case).                                 !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
cdef float FirstMom_P_logMHj_all_c( float logMs,  params_LT,  params_ET, float logR1, float logR2, float flag_Hj):
    cdef int N=510
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy, logMHj_dummy
    s = (N+1)*[0]

    delta_logR = (logR2 - logR1) / (N * 1.0)

    #Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        logMHj_dummy = logR_dummy + logMs
        s[i] = P_MHj_Ms_all(logR_dummy + logMs, logMs, params_LT, params_ET, flag_Hj) * logMHj_dummy

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2))<1E-5):
            break

        integral2=integral1

    logMHj_dummy = logR2 + logMs
    s[N] = P_MHj_Ms_all(logR2 + logMs, logMs, params_LT, params_ET, flag_Hj) * logMHj_dummy
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!

def FirstMom_P_logMHj_all(  float logMs,  params_LT,  params_ET, float logR1, float logR2, float flag_Hj):
    return FirstMom_P_logMHj_all_c( logMs,  params_LT,  params_ET, logR1, logR2, flag_Hj)

#=======================================================================!
# FUNCTION:  SecondMom_P_logMHj_all_c                                    !
#   This function computes the Second Moment of the logRHI and logRH2    !
#   distributions (LTGs case).                                           !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
cdef float SecondMom_P_logMHj_all_c( float logMs,  params_LT,  params_ET,  float logR1,  float logR2,  float flag_Hj):
    cdef int N=510
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy, Term_prod1, logMHj_ave, logMHj_dummy
    s = (N+1)*[0.0]

    delta_logR = (logR2 - logR1) / N
    logMHj_ave = FirstMom_P_logMHj_all_c(logMs, params_LT, params_ET, logR1, logR2, flag_Hj)
    
    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        logMHj_dummy = logR_dummy + logMs 
        
        Term_prod1 = logMHj_dummy - logMHj_ave
        Term_prod1 = Term_prod1 * Term_prod1
        s[i] = P_MHj_Ms_all(logR_dummy + logMs, logMs, params_LT, params_ET, flag_Hj) * Term_prod1

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2))<1E-5):
            break

        integral2=integral1

    logMHj_dummy = logR2 + logMs 
    Term_prod1 = logMHj_dummy - logMHj_ave
    Term_prod1 = Term_prod1 * Term_prod1
    s[N] = P_MHj_Ms_all(logR2 + logMs, logMs, params_LT, params_ET, flag_Hj) * Term_prod1
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

def SecondMom_P_logMHj_all( logMs,  params_LT,  params_ET,  logR1,  logR2,  flag_Hj):
    return SecondMom_P_logMHj_all_c( logMs,  params_LT,  params_ET,  logR1,  logR2,  flag_Hj)

#=======================================================================!
# FUNCTION:  FirstMom_P_RHj_LTGs                                   !
#   This function computes the First Moment of the logRHI and logRH2     !
#   Schechter distributions (LTGs case).                                 !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
cdef float FirstMom_P_RHj_LTGs( float logMs,  params,  float logR1,  float logR2,  float flag_Hj):
    cdef int N=510
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy
    s = (N+1)*[0.0]

    delta_logR = (logR2 - logR1) / N

    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        s[i] = P_MHj_Ms_blue_LTGs(logR_dummy + logMs, logMs, params, flag_Hj) * np.power(10, logR_dummy)

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2)) < 1E-5):
            break

        integral2=integral1

    s[N] = P_MHj_Ms_blue_LTGs(logR2 + logMs, logMs, params, flag_Hj) * np.power(10, logR2)
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
def FM_P_RHj_LTGs( float logMs,  params,  float logR1,  float logR2,  float flag_Hj ):
    return FirstMom_P_RHj_LTGs( logMs,  params,  logR1,  logR2,  flag_Hj)

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
cdef float SecondMom_P_RHj_LTGs( float logMs, params,  float logR1,  float logR2,  float flag_Hj):
    cdef int N=710
    cdef integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy, R_ave
    cdef float Term_prod1, 
    s = (N+1)*[0]

    delta_logR = (logR2 - logR1) / (N * 1.0)
    R_ave = FirstMom_P_RHj_LTGs(logMs, params, logR1, logR2, flag_Hj)

    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)

        Term_prod1 = np.power(10, logR_dummy) - R_ave
        Term_prod1 = Term_prod1 * Term_prod1
        s[i] = P_MHj_Ms_blue_LTGs(logR_dummy + logMs, logMs, params, flag_Hj) * Term_prod1

        if(i>=1 and i<=N-1):
            integral1+=s[i]
        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2)) < 1E-5):
            break

        integral2=integral1

    Term_prod1 = np.power(10, logR2) - R_ave
    Term_prod1 = Term_prod1 * Term_prod1
    s[N] = P_MHj_Ms_blue_LTGs(logR2 + logMs, logMs, params, flag_Hj) * Term_prod1
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
def SM_P_RHj_LTGs( float logMs,  params,  float logR1,  float logR2,  float flag_Hj ):
    return SecondMom_P_RHj_LTGs(  logMs, params,  logR1,  logR2,  flag_Hj )

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
cdef float FirstMom_P_RHj_ETGs( float logMs, params, float logR1, float logR2, float flag_Hj):
    cdef int N=510
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy
    s = (N+1)*[0.0]

    delta_logR = (logR2 - logR1) / (N * 1.0)

    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        s[i] = P_MHj_Ms_red_ETGs(logR_dummy + logMs, logMs, params, flag_Hj) * np.power(10, logR_dummy)

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2))<1E-5):
            integral2=integral1

    s[N] = P_MHj_Ms_red_ETGs(logR2 + logMs, logMs, params, flag_Hj) * np.power(10, logR2)
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
def FM_P_RHj_ETGs( float logMs, params, float logR1, float logR2, float flag_Hj ):
    return FirstMom_P_RHj_ETGs( logMs, params, logR1, logR2, flag_Hj )

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
cdef float SecondMom_P_RHj_ETGs( float logMs, params, float logR1, float logR2, float flag_Hj):
    cdef int N=510
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy, Term_prod1, R_ave
    s = (N+1)*[0.0]

    delta_logR = (logR2 - logR1) / N
    R_ave = FirstMom_P_RHj_ETGs(logMs, params, logR1, logR2, flag_Hj)
    
    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        Term_prod1 = np.power(10, logR_dummy) - R_ave
        Term_prod1 = Term_prod1 * Term_prod1
        s[i] = P_MHj_Ms_red_ETGs(logR_dummy + logMs, logMs, params, flag_Hj) * Term_prod1

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2))<1E-5):
            break

        integral2=integral1


    Term_prod1 = np.power(10, logR2) - R_ave
    Term_prod1 = Term_prod1 * Term_prod1
    s[N] = P_MHj_Ms_red_ETGs(logR2 + logMs, logMs, params, flag_Hj) * Term_prod1
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#def test(float logMs, float slope, float b):
    #return x2_P_TH_dlogR(logMs, slope, b) + F_P_TH_dlogR(logMs, slope, b) + logRstr(logMs, 1.05, -1.2, 0.54, 0.02) + phistr(0.5) + alpha_logMs(logMs, slope, b) - Sx_logx(-0.5, 0.52, -1.2)

#=======================================================================!
def SM_P_RHj_ETGs( float logMs, params, float logR1, float logR2, float flag_Hj ):
    return SecondMom_P_RHj_ETGs( logMs, params, logR1, logR2, flag_Hj )

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
def FirstMom_P_RHj_all(  float logMs,  params_LT,  params_ET, float logR1, float logR2, float flag_Hj):
#cdef float FirstMom_P_RHj_all( float logMs,  params_LT,  params_ET, float logR1, float logR2, float flag_Hj):
    cdef int N=510
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy
    s = (N+1)*[0]

    delta_logR = (logR2 - logR1) / (N * 1.0)

    #Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        s[i] = P_MHj_Ms_all(logR_dummy + logMs, logMs, params_LT, params_ET, flag_Hj) * np.power(10, logR_dummy)

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2))<1E-5):
            break

        integral2=integral1

    s[N] = P_MHj_Ms_all(logR2 + logMs, logMs, params_LT, params_ET, flag_Hj) * np.power(10, logR2)
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
#cdef float SecondMom_P_RHj_all( float logMs,  params_LT,  params_ET,  float logR1,  float logR2,  float flag_Hj):
    cdef int N=710
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy, Term_prod1, R_ave
    s = (N+1)*[0.0]

    delta_logR = (logR2 - logR1) / N
    R_ave = FirstMom_P_RHj_all(logMs, params_LT, params_ET, logR1, logR2, flag_Hj)

    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        Term_prod1 = np.power(10, logR_dummy) - R_ave
        Term_prod1 = Term_prod1 * Term_prod1
        s[i] = P_MHj_Ms_all(logR_dummy + logMs, logMs, params_LT, params_ET, flag_Hj) * Term_prod1

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2))<1E-5):
            break

        integral2=integral1

    Term_prod1 = np.power(10, logR2) - R_ave
    Term_prod1 = Term_prod1 * Term_prod1
    s[N] = P_MHj_Ms_all(logR2 + logMs, logMs, params_LT, params_ET, flag_Hj) * Term_prod1
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
# FUNCTION:  FirstMom_Schech_MHj_LTGs                                   !
#   This function computes the First Moment of the logRHI and logRH2     !
#   Schechter distributions (LTGs case).                                 !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
cdef float FirstMom_Schech_MHj_LTGs( float logMs,  params,  float logR1,  float logR2,  float flag_Hj):
    cdef int N=510
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy, MHj_dummy
    s = (N+1)*[0.0]

    delta_logR = (logR2 - logR1) / N

    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        MHj_dummy = np.power(10, logR_dummy + logMs)
        s[i] = P_MHj_Ms_blue_LTGs(logR_dummy + logMs, logMs, params, flag_Hj) * MHj_dummy

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2)) < 1E-5):
            break

        integral2=integral1

    s[N] = P_MHj_Ms_blue_LTGs(logR2 + logMs, logMs, params, flag_Hj) * np.power(10, logR2 + logMs)
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
def FM_MHj_LTGs( float logMs,  params,  float logR1,  float logR2,  float flag_Hj ):
    return FirstMom_Schech_MHj_LTGs( logMs,  params,  logR1,  logR2,  flag_Hj)

#=======================================================================!
# FUNCTION:  SecondMom_P_MHj_LTGs                                        !
#   This function computes the Second Moment of the logRHI and logRH2    !
#   distributions (LTGs case).                                           !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
cdef float SecondMom_P_MHj_LTGs( float logMs, params,  float logR1,  float logR2,  float flag_Hj):
    cdef int N=710
    cdef integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy, logMHj_dummy
    cdef float Term_prod1, 
    s = (N+1)*[0]

    delta_logR = (logR2 - logR1) / (N * 1.0)

    # Integrating function */
    log_MHj_ave = np.log10(FirstMom_Schech_MHj_LTGs(logMs, params, logR1, logR2, flag_Hj))

    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        logMHj_dummy = logR_dummy + logMs
        
        #Term_prod1 = logMHj_dummy - np.log10(1 + np.power(10, log_MHj_ave - logMHj_dummy))
        #Term_prod1 = np.power(10, 2 * Term_prod1)
        Term_prod1 = np.power(np.power(10, logMHj_dummy) - np.power(10, log_MHj_ave), 2)
        s[i] = P_MHj_Ms_blue_LTGs(logR_dummy + logMs, logMs, params, flag_Hj) * Term_prod1

        if(i>=1 and i<=N-1):
            integral1+=s[i]
        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2)) < 1E-5):
            break

        integral2=integral1
    
    logMHj_dummy = logR2 + logMs
    log_MHj_ave = np.log10(FirstMom_Schech_MHj_LTGs(logMs, params, logR1, logR2, flag_Hj))
    
    #Term_prod1 = logMHj_dummy - np.log10(1 + np.power(10, log_MHj_ave - logMHj_dummy))
    #Term_prod1 = np.power(10, 2 * Term_prod1)
    Term_prod1 = np.power(np.power(10, logMHj_dummy) - np.power(10, log_MHj_ave), 2)
    s[N] = P_MHj_Ms_blue_LTGs(logR2 + logMs, logMs, params, flag_Hj) * Term_prod1
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
def SM_MHj_LTGs( float logMs, params,  float logR1,  float logR2,  float flag_Hj ):
    return SecondMom_P_MHj_LTGs( logMs, params,  logR1,  logR2,  flag_Hj)

#=======================================================================!
# FUNCTION:  FirstMom_Schech_MHj_ETGs                                    !
#   This function computes the First Moment of the logRHI and logRH2     !
#   Schechter distributions (LTGs case).                                 !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
cdef float FirstMom_Schech_MHj_ETGs( float logMs, params, float logR1, float logR2, float flag_Hj):
    cdef int N=510
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy, logMHj_dummy
    s = (N+1)*[0.0]

    delta_logR = (logR2 - logR1) / (N * 1.0)

    # Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        logMHj_dummy = logR_dummy + logMs
        s[i] = P_MHj_Ms_red_ETGs(logR_dummy + logMs, logMs, params, flag_Hj) * np.power(10, logMHj_dummy)

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2))<1E-5):
            integral2=integral1

    s[N] = P_MHj_Ms_red_ETGs(logR2 + logMs, logMs, params, flag_Hj) * np.power(10, logR2 + logMs)
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
def FM_P_MHj_ETGs( float logMs, params, float logR1, float logR2, float flag_Hj ):
    return FirstMom_Schech_MHj_ETGs( logMs, params, logR1, logR2, flag_Hj )

#=======================================================================!
# FUNCTION:  SecondMom_P_MHj_ETGs                                        !
#   This function computes the Second Moment of the logRHI and logRH2    !
#   distributions (LTGs case).                                           !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
cdef float SecondMom_P_MHj_ETGs( float logMs, params, float logR1, float logR2, float flag_Hj):
    cdef int N=1000
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy, Term_prod1, logMHj_dummy, logMHj_ave
    s = (N+1)*[0.0]

    delta_logR = (logR2 - logR1) / N
    
    # Integrating function */
    logMHj_ave = np.log10(FirstMom_Schech_MHj_ETGs(logMs, params, logR1, logR2, flag_Hj))

    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        logMHj_dummy = logR_dummy + logMs
        
        Term_prod1 = np.power(np.power(10, logMHj_dummy) - np.power(10, logMHj_ave), 2)
        s[i] = P_MHj_Ms_red_ETGs(logR_dummy + logMs, logMs, params, flag_Hj) * Term_prod1

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2))<1E-5):
            break

        integral2=integral1


    logMHj_dummy = logR2 + logMs

    Term_prod1 = np.power( np.power(10, logMHj_dummy) - np.power(10, logMHj_ave), 2)
    s[N] = P_MHj_Ms_red_ETGs(logR2 + logMs, logMs, params, flag_Hj) * Term_prod1
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#def test(float logMs, float slope, float b):
    #return x2_P_TH_dlogR(logMs, slope, b) + F_P_TH_dlogR(logMs, slope, b) + logRstr(logMs, 1.05, -1.2, 0.54, 0.02) + phistr(0.5) + alpha_logMs(logMs, slope, b) - Sx_logx(-0.5, 0.52, -1.2)

#=======================================================================!
def SM_P_MHj_ETGs( float logMs, params, float logR1, float logR2, float flag_Hj ):
    return SecondMom_P_MHj_ETGs( logMs, params, logR1, logR2, flag_Hj )

#=======================================================================!
# FUNCTION:  FirstMom_P_MHj_all_c                                         !
#   This function computes the First Moment of the logRHI and logRH2     !
#   Schechter distributions (LTGs case).                                 !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
cdef float FirstMom_P_MHj_all_c( float logMs,  params_LT,  params_ET, float logR1, float logR2, float flag_Hj):
    cdef int N=510
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy, logMHj_dummy
    s = (N+1)*[0]

    delta_logR = (logR2 - logR1) / (N * 1.0)
    
    
    #Integrating function */
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        logMHj_dummy = logR_dummy + logMs
        
        s[i] = P_MHj_Ms_all(logR_dummy + logMs, logMs, params_LT, params_ET, flag_Hj) * np.power(10, logMHj_dummy)

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2))<1E-5):
            break

        integral2=integral1

    logMHj_dummy = logR2 + logMs
    s[N] = P_MHj_Ms_all(logR2 + logMs, logMs, params_LT, params_ET, flag_Hj) * np.power(10, logMHj_dummy)
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

#=======================================================================!
def FirstMom_P_MHj_all( float logMs,  params_LT,  params_ET, float logR1, float logR2, float flag_Hj):
    return FirstMom_P_MHj_all_c( logMs,  params_LT,  params_ET, logR1, logR2, flag_Hj)

#=======================================================================!
# FUNCTION:  SecondMom_P_MHj_all_c                                       !
#   This function computes the Second Moment of the logRHI and logRH2    !
#   distributions (LTGs case).                                           !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
cdef float SecondMom_P_MHj_all_c( float logMs,  params_LT,  params_ET,  float logR1,  float logR2,  float flag_Hj):
    cdef int N=810
    cdef float integral=0.0, integral1=0.0, integral2=0.0, delta_logR, logR_dummy, Term_prod1, logMHj_dummy, logMHj_ave
    s = (N+1)*[0.0]

    delta_logR = (logR2 - logR1) / N

    # Integrating function */
    logMHj_ave = np.log10(FirstMom_P_MHj_all_c(logMs, params_LT, params_ET, logR1, logR2, flag_Hj))
    
    for i in range(N+1):
        logR_dummy = logR1 + (delta_logR * i)
        logMHj_dummy = logR_dummy + logMs
        Term_prod1 = np.power(np.power(10, logMHj_dummy) - np.power(10, logMHj_ave), 2)
        
        s[i] = P_MHj_Ms_all(logR_dummy + logMs, logMs, params_LT, params_ET, flag_Hj) * Term_prod1

        if(i>=1 and i<=N-1):
            integral1+=s[i]

        if(np.sqrt(np.power(np.log10(integral1) - np.log10(integral2),2))<1E-5):
            break

        integral2=integral1
    
    logMHj_dummy = logR2 + logMs 
    Term_prod1 = np.power(np.power(10, logMHj_dummy) - np.power(10, logMHj_ave), 2)

    s[N] = P_MHj_Ms_all(logR2 + logMs, logMs, params_LT, params_ET, flag_Hj) * Term_prod1
    integral = 0.5 * delta_logR * (s[0] + (2 * integral1) + s[N])

    return integral

def SecondMom_P_MHj_all( float logMs,  params_LT,  params_ET,  float logR1,  float logR2,  float flag_Hj):
    return SecondMom_P_MHj_all_c( logMs,  params_LT,  params_ET,  logR1,  logR2,  flag_Hj)

#=======================================================================!
# FUNCTION:  percentiles_logRHj_LTGs                                     !
#   Computes percentiles of cold gas conditional distributions.          !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
def percentiles_logRHj_LTGs( logMs,  params,  logR1,  logR2,  flag_Hj, p1, p2, p3):
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
    
    results = []
    results.append(per_left)
    results.append(per_mid)
    results.append(per_right)

    return results

#========================================================================!
# FUNCTION:  percentiles_RHj_LTGs                                        !
#   Computes percentiles of cold gas conditional distributions.          !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
def percentiles_RHj_LTGs( logMs,  params,  logR1,  logR2,  flag_Hj, p1, p2, p3):
    return np.power(10, percentiles_logRHj_LTGs( logMs,  params,  logR1,  logR2,  flag_Hj, p1, p2, p3) )
    
#=======================================================================!
# FUNCTION:  percentiles_logRHj_ETGs                                     !
#   Computes percentiles of cold gas conditional distributions.          !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
def percentiles_logRHj_ETGs( logMs,  params,  logR1,  logR2,  flag_Hj, p1, p2, p3):
    x1=0
    Nbins=400
    Delta = (logR2 - logR1) / Nbins

    for j in range (1,4):
        if(j == 1):
            Prandom = float(p1)
        elif (j == 2):
            Prandom = float(p2)
        elif (j == 3):
            Prandom = float(p3)

        for i in range(Nbins + 1):
            logRFalse = (logR1) + Delta*i
            Pcumu = float(Pcumu_Schech_logR_ETGs(logMs, params, logRFalse, logR2, flag_Hj))
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

    results = []
    results.append(per_left)
    results.append(per_mid)
    results.append(per_right)

    return results

#=======================================================================!
# FUNCTION:  percentiles_logRHj_all                                     !
#   Computes percentiles of cold gas conditional distributions.          !
#                                                                        !
#                                                                        !
#                                                                        !
#  Notes:                                                                !
#                                                                        !
#                                                                        !
#=======================================================================*/
def percentiles_logRHj_all( logMs,  params_LTGs, params_ETGs,  logR1,  logR2,  flag_Hj, p1, p2, p3):
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
            Pcumu = Pcumu_Schech_logR_all_c(logMs, params_LTGs, params_ETGs, logRFalse, logR2, flag_Hj)
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

            Pcumu = Pcumu_Schech_logR_all_c(logMs, params_LTGs, params_ETGs, Xm, logR2, flag_Hj)
            Pcumu2 = Pcumu_Schech_logR_all_c(logMs, params_LTGs, params_ETGs, x1, logR2, flag_Hj)

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
    
    results = []
    results.append(per_left)
    results.append(per_mid)
    results.append(per_right)

    return results

