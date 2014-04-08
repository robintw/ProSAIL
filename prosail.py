#!/usr/bin/python
# coding=utf-8
import os
import numpy as np
from numpy import pi

class Prosail:

    # Python version of PROSAIL model translated from Fortran and IDL versions by: 
    # Guy Serbin, Spatial Analysis Unit, REDP, Teagasc, Ashtown, Dublin 15, Ireland
    # Email: guy.serbin 'at' teagasc.ie or gmail.com
    # This version requires the numpy package from http://www.numpy.org/

    ## Soil reflectance

    def _soilref(self, psoil, Rsoil1, Rsoil2):
        """Linear soil wetness mixing model"""

        rsoil0=psoil*Rsoil1+(1-psoil)*Rsoil2
        return rsoil0

    def _tav_abs(self, theta,refr):
        """Computation of the average transmittivity at the leaf surface within a given
        solid angle. teta is the incidence solid angle (in radian). The average angle
        that works in most cases is 40deg*pi/180. ref is the refaction index.

        Stern F. (1964), Transmission of isotropic radiation across an interface between
        two dielectrics, Applied Optics, 3:111-113.
        Allen W.A. (1973), Transmission of isotropic light across a dielectric surface in
        two and three dimensions, Journal of the Optical Society of America, 63:664-666.
        """

        refr=np.array(refr)
        thetarad=np.radians(theta)
        res=np.zeros(refr.size)
        if (theta == 0.):
            res=4.*refr/(refr+1.)**2
        else:
            refr2=refr*refr
            ax=(refr+1.)**2/2.
            bx=-(refr2-1.)**2/4.
            
            if (thetarad == pi/2.):
                b1=0.
            else:
                b1=((np.sin(thetarad)**2-(refr2+1.)/2.)**2+bx)**0.5
            b2=np.sin(thetarad)**2-(refr2+1.)/2.
            b0=b1-b2
            ts=(bx**2/(6.*b0**3)+bx/b0-b0/2.)-(bx**2/(6.*ax**3)+bx/ax-ax/2.)
            tp2=np.zeros(refr.size)
            tp4=np.zeros(refr.size)
            tp1=-2.*refr2*(b0-ax)/(refr2+1.)**2
            tp2=-2.*refr2*(refr2+1.)*np.log(b0/ax)/(refr2-1.)**2
            tp3=refr2*(1./b0-1./ax)/2.
            tp4=16.*refr2**2*(refr2**2+1.)*np.log((2.*(refr2+1.)*b0-(refr2-1.)**2)/ 
                    (2.*(refr2+1.)*ax-(refr2-1.)**2))/((refr2+1.)**3*(refr2-1.)**2)
            tp5=16.*refr2**3*(1./(2.*(refr2+1.)*b0-((refr2-1.)**2))-1./(2.*(refr2+1.) 
                *ax-(refr2-1.)**2))/(refr2+1.)**3
            tp=tp1+tp2+tp3+tp4+tp5
            res=(ts+tp)/(2.*np.sin(thetarad)**2)
        
        return res

    def _prospect_5B(self, N,Cab,Car,Cbrown,Cw,Cm,spectra):
        """PROSPECT model, by Jean-Baptiste Feret & Stephane Jacquemoud

        Féret J.B., François C., Asner G.P., Gitelson A.A., Martin R.E., Bidel L.P.R.,
        Ustin S.L., le Maire G., Jacquemoud S. (2008), PROSPECT-4 and 5: Advances in the
        leaf optical properties model separating photosynthetic pigments, Remote Sennp.sing
        of Environment, 112:3030-3043.
        Jacquemoud S., Ustin S.L., Verdebout J., Schmuck G., Andreoli G., Hosgood B.
        (1996), Estimating leaf biochemistry unp.sing the PROSPECT leaf optical properties
        model, Remote Sennp.sing of Environment, 56:194-202.
        Jacquemoud S., Baret F. (1990), PROSPECT: a model of leaf optical properties
        spectra, Remote Sensing of Environment, 34:75-91.
        """

        k=(Cab*np.array(spectra[2])+Car*np.array(spectra[3])+Cbrown*np.array(spectra[4])+Cw*np.array(spectra[5])+Cm*np.array(spectra[6]))/N
        refractive=np.array(spectra[1])
        # ********************************************************************************
        # reflectance and transmittance of one layer
        # ********************************************************************************
        # Allen W.A., Gausman H.W., Richardson A.J., Thomas J.R. (1969), Interaction of
        # isotropic ligth with a compact plant leaf, Journal of the Optical Society of
        # American, 59:1376-1379.
        # ********************************************************************************
        
        # np.exponential integral: S13AAF routine from the NAG library
        tau=np.zeros(k.size)
        xx=np.zeros(k.size)
        yy=np.zeros(k.size)
        
        for i in range(tau.size):
            if k[i]<=0.0:
                tau[i]=1
            elif (k[i]>0.0 and k[i]<=4.0):
                xx[i]=0.5*k[i]-1.0
                yy[i]=(((((((((((((((-3.60311230482612224e-13 
                    *xx[i]+3.46348526554087424e-12)*xx[i]-2.99627399604128973e-11) 
                    *xx[i]+2.57747807106988589e-10)*xx[i]-2.09330568435488303e-9) 
                    *xx[i]+1.59501329936987818e-8)*xx[i]-1.13717900285428895e-7) 
                    *xx[i]+7.55292885309152956e-7)*xx[i]-4.64980751480619431e-6) 
                    *xx[i]+2.63830365675408129e-5)*xx[i]-1.37089870978830576e-4) 
                    *xx[i]+6.47686503728103400e-4)*xx[i]-2.76060141343627983e-3) 
                    *xx[i]+1.05306034687449505e-2)*xx[i]-3.57191348753631956e-2) 
                    *xx[i]+1.07774527938978692e-1)*xx[i]-2.96997075145080963e-1
                yy[i]=(yy[i]*xx[i]+8.64664716763387311e-1)*xx[i]+7.42047691268006429e-1
                yy[i]=yy[i]-np.log(k[i])
                tau[i]=(1.0-k[i])*np.exp(-k[i])+k[i]**2*yy[i]
            elif (k[i]>4.0 and k[i]<=85.0):
                xx[i]=14.5/(k[i]+3.25)-1.0
                yy[i]=(((((((((((((((-1.62806570868460749e-12 
                    *xx[i]-8.95400579318284288e-13)*xx[i]-4.08352702838151578e-12) 
                    *xx[i]-1.45132988248537498e-11)*xx[i]-8.35086918940757852e-11) 
                    *xx[i]-2.13638678953766289e-10)*xx[i]-1.10302431467069770e-9) 
                    *xx[i]-3.67128915633455484e-9)*xx[i]-1.66980544304104726e-8) 
                    *xx[i]-6.11774386401295125e-8)*xx[i]-2.70306163610271497e-7) 
                    *xx[i]-1.05565006992891261e-6)*xx[i]-4.72090467203711484e-6) 
                    *xx[i]-1.95076375089955937e-5)*xx[i]-9.16450482931221453e-5) 
                    *xx[i]-4.05892130452128677e-4)*xx[i]-2.14213055000334718e-3
                yy[i]=((yy[i]*xx[i]-1.06374875116569657e-2)*xx[i]-8.50699154984571871e-2)*xx[i]+9.23755307807784058e-1
                yy[i]=np.exp(-k[i])*yy[i]/k[i]
                tau[i]=(1.0-k[i])*np.exp(-k[i])+k[i]**2*yy[i]
            else:
                tau[i]=0
        
        # transmissivity of the layer
        
        theta1=90.
        t1= self._tav_abs(theta1,refractive)
        theta2=40.
        t2= self._tav_abs(theta2,refractive)
        x1=1-t1
        x2=t1**2*tau**2*(refractive**2-t1)
        x3=t1**2*tau*refractive**2
        x4=refractive**4-tau**2*(refractive**2-t1)**2
        x5=t2/t1
        x6=x5*(t1-1)+1-t2
        r=x1+x2/x4
        t=x3/x4
        ra=x5*r+x6
        ta=x5*t
        
        # ********************************************************************************
        # reflectance and transmittance of N layers
        # ********************************************************************************
        # Stokes G.G. (1862), On the intensity of the light reflected from or transmitted
        # through a pile of plates, Proceedings of the Royal Society of London, 11:545-556.
        # ********************************************************************************
        
        delta=(t**2-r**2-1)**2-4*r**2
        beta=(1+r**2-t**2-delta**0.5)/(2*r)
        va=(1+r**2-t**2+delta**0.5)/(2*r)
        vb=(beta*(va-r)/(va*(beta-r)))**0.5
        s1=ra*(va*vb**(N-1)-va**(-1)*vb**(-(N-1)))+(ta*t-ra*r)*(vb**(N-1)-vb**(-(N-1)))
        s2=ta*(va-va**(-1))
        s3=va*vb**(N-1)-va**(-1)*vb**(-(N-1))-r*(vb**(N-1)-vb**(-(N-1)))
        RN=s1/s3
        TN=s2/s3
        return RN, TN

    def _calcLidf(self, TypeLidf=1,LIDFa=-0.35,LIDFb=-0.15):
        if (type(TypeLidf)==type('')) or ((TypeLidf !=1) and (TypeLidf !=2)):
            if (type(TypeLidf)==type('')):
                if TypeLidf.lower()=='Planophile'.lower():
                    LIDFa=1
                    LIDFb=0
                elif TypeLidf.lower()=='Erectophile'.lower():
                    LIDFa=-1
                    LIDFb=0
                elif TypeLidf.lower()=='Plagiophile'.lower():
                    LIDFa=0
                    LIDFb=-1
                elif TypeLidf.lower()=='Extremophile'.lower():
                    LIDFa=0
                    LIDFb=1
                elif TypeLidf.lower()=='Spherical'.lower():
                    LIDFa=-0.35
                    LIDFb=-0.15
            else: # Assuming uniform leaf distribution here
                print('Warning: Uniform leaf distribution chosen. If this was not your intent, double-check to make sure you used the proper parameters.')
                LIDFa=0
                LIDFb=0
            TypeLidf=1
        #	Generate leaf angle distribution from average leaf angle (ellipsoidal) or (a,b) parameters
        if(TypeLidf==1):
            lidf= self._dladgen(LIDFa,LIDFb)
        elif(TypeLidf==2):
            na=13
            lidf=self._calc_LIDF_ellipsoidal(na,LIDFa)
        return lidf

    def _dladgen(self, a,b):
        t=np.zeros(13)
        freq=np.zeros(13)
        for i in range(13):
            if i<=7:
                t[i]=(i+1)*10. 
                freq[i]=self._dcum(a,b,t[i])
            elif i>=8 and i<12:
                t[i]=80.+(i-7)*2.
                freq[i]=self._dcum(a,b,t[i])
            else:
                freq[i]=1.
        for i in reversed(range(13)):
            if i>=1:
                freq[i]=freq[i]-freq[i-1]
        return freq

    def _dcum(self, a,b,t):
        if (a>1.):
            dcum=1.-np.cos(np.radians(t))
        else:
            eps=1e-8
            delx=1.
            x=2*np.radians(t)
            p=x
            while (delx>eps):
                y = a*np.sin(x)+.5*b*np.sin(2.*x)
                dx=.5*(y-x+p)
                x=x+dx
                delx=np.absolute(dx)
            dcum=(2.*y+p)/pi
        return dcum

    #********************************************************************************
    #*                          Campbell.f                            
    #*     
    #*    Computation of the leaf angle distribution function value (freq) 
    #*    Ellipsoidal distribution function caracterised by the average leaf 
    #*    inclination angle in degree (ala)                                     
    #*    Campbell 1986                                                      
    #*                                                                              
    #********************************************************************************

    def _campbell(self, n,ala):
        """
        Computation of the leaf angle distribution function value (freq) 
        Ellipsoidal distribution function caracterised by the average leaf 
        inclination angle in degree (ala)                                  
        Campbell 1986                                                      
        """

        tx2=np.array([0., 10., 20., 30., 40., 50., 60., 70., 80., 82., 84., 86., 88.])
        tx1=np.array([10., 20., 30., 40., 50., 60., 70., 80., 82., 84., 86., 88., 90.])
        
        tl1=tx1*np.arctan(1.)/45.
        tl2=tx2*np.arctan(1.)/45.
        excent=np.exp(-1.6184e-5*ala**3+2.1145e-3*ala**2-1.2390e-1*ala+3.2491)
        x1  = excent/(np.sqrt(1.+excent**2*np.tan(tl1)**2))
        x2  = excent/(np.sqrt(1.+excent**2*np.tan(tl2)**2))
        if (excent==1.):
            freq = np.absolute(np.cos(tl1)-np.cos(tl2))
        else:
            alpha  = excent/np.sqrt(np.absolute(1.-excent**2))
            alpha2 = alpha**2
            x12 = x1**2
            x22 = x2**2
            if (excent>1):
                alpx1 = np.sqrt(alpha2+x12)
                alpx2 = np.sqrt(alpha2+x22)
                dum   = x1*alpx1+alpha2*np.log(x1+alpx1)
                freq  = np.absolute(dum-(x2*alpx2+alpha2*np.log(x2+alpx2)))
            else:
                almx1 = np.sqrt(alpha2-x12)
                almx2 = np.sqrt(alpha2-x22)
                dum   = x1*almx1+alpha2*np.arcsin(x1/alpha)
                freq  = np.absolute(dum-(x2*almx2+alpha2*np.arcsin(x2/alpha)))
        sum0 = np.sum(freq)
        freq=freq/sum0	#*100.

        return freq

    def _calc_LIDF_ellipsoidal(self, na,alpha):
        freq= self._campbell(na,alpha)
        return freq

    def _volscatt(self, tts,tto,psi,ttl):
        """
        tts     = solar zenith
        tto     = viewing zenith
        psi     = azimuth
        ttl     = leaf inclination angle
        chi_s   = interception functions
        chi_o   = interception functions
        frho    = function to be multiplied by leaf reflectance rho
        ftau    = functions to be multiplied by leaf transmittance tau
        """
        #	Compute volume scattering functions and interception coefficients
        #	for given solar zenith, viewing zenith, azimuth and leaf inclination angle.
        
        #	chi_s and chi_o are the interception functions.
        #	frho and ftau are the functions to be multiplied by leaf reflectance rho and
        #	leaf transmittance tau, respectively, in order to obtain the volume scattering
        #	function.
        
        #	Wout Verhoef, april 2001, for CROMA
        rd=pi/180.
        costs=np.cos(rd*tts)
        costo=np.cos(rd*tto)
        sints=np.sin(rd*tts)
        sinto=np.sin(rd*tto)
        cospsi=np.cos(rd*psi)
        
        psir=rd*psi
        
        costl=np.cos(rd*ttl)
        sintl=np.sin(rd*ttl)
        cs=costl*costs
        co=costl*costo
        ss=sintl*sints
        so=sintl*sinto
        
        #c ..............................................................................
        #c     betas -bts- and betao -bto- computation
        #c     Transition angles (beta) for solar (betas) and view (betao) directions
        #c     if thetav+thetal>pi/2, bottom side of the leaves is observed for leaf azimut 
        #c     interval betao+phi<leaf azimut<2pi-betao+phi.
        #c     if thetav+thetal<pi/2, top side of the leaves is always observed, betao=pi
        #c     same consideration for solar direction to compute betas
        #c ..............................................................................

        cosbts=5.
        if (np.absolute(ss)>1e-6):
            cosbts=-cs/ss
        
        cosbto=5.
        if (np.absolute(so)>1e-6):
            cosbto=-co/so
        
        if (np.absolute(cosbts)<1.):
            bts=np.arccos(cosbts)
            ds=ss
        else:
            bts=pi
            ds=cs
        
        chi_s=2./pi*((bts-pi*.5)*cs+np.sin(bts)*ss)
        
        if (np.absolute(cosbto)<1.):
            bto=np.arccos(cosbto)
            doo=so
        elif(tto<90.):
            bto=pi
            doo=co
        else:
            bto=0
            doo=-co
        
        chi_o=2./pi*((bto-pi*.5)*co+np.sin(bto)*so)
        
        #c ..............................................................................
        #c   Computation of auxiliary azimut angles bt1, bt2, bt3 used          
        #c   for the computation of the bidirectional scattering coefficient w              
        #c .............................................................................

        btran1=np.absolute(bts-bto)
        btran2=pi-np.absolute(bts+bto-pi)
        
        if (psir<=btran1):
            bt1=psir
            bt2=btran1
            bt3=btran2
        else:
            bt1=btran1
            if (psir<=btran2):
                bt2=psir
                bt3=btran2
            else:
                bt2=btran2
                bt3=psir
        
        t1=2.*cs*co+ss*so*cospsi
        t2=0.
        if (bt2>0.):
            t2=np.sin(bt2)*(2.*ds*doo+ss*so*np.cos(bt1)*np.cos(bt3))
        
        denom=2.*pi*pi
        frho=((pi-bt2)*t1+t2)/denom
        ftau=    (-bt2*t1+t2)/denom
        
        if (frho<0):
            frho=0
        
        if (ftau<0):
            ftau=0
        
        return chi_s,chi_o,frho,ftau
        
    def _PRO4SAIL(self, rho,tau,lidf,lai,q,tts,tto,psi,rsoil):
        """
        This version has been implemented by Jean-Baptiste Féret
        Jean-Baptiste Féret takes the entire responsibility for this version 
        All comments, changes or questions should be sent to:
        jbferet@stanford.edu

        Jean-Baptiste Féret
        Institut de Physique du Globe de Paris
        Space and Planetary Geophysics
        October 2009
        this model PRO4SAIL is based on a version provided by
        Wout Verhoef 
        NLR 
        April/May 2003,
        original version downloadable at http://teledetection.ipgp.jussieu.fr/prosail/
        Improved and extended version of SAILH model that avoids numerical singularities
        and works more efficiently if only few parameters change.
        ferences:
        Verhoef et al. (2007) Unified Optical-Thermal Four-Stream Radiative
        Transfer Theory for Homogeneous Vegetation Canopies, IEEE TRANSACTIONS 
        ON GEOSCIENCE AND REMOTE SENSING, VOL. 45, NO. 6, JUNE 2007
        """

        litab=np.array([5.,15.,25.,35.,45.,55.,65.,75.,81.,83.,85.,87.,89.])
        
        rd=pi/180.
        
        #if(flag[1]):
        #	Geometric quantities
        cts		= np.cos(rd*tts)
        cto		= np.cos(rd*tto)
        ctscto	= cts*cto
        tants	= np.tan(rd*tts)
        tanto	= np.tan(rd*tto)
        cospsi	= np.cos(rd*psi)
        dso		= np.sqrt(tants*tants+tanto*tanto-2.*tants*tanto*cospsi)

        na=13
        #if (flag[2]):
        

        # angular distance, compensation of shadow length
        #if (flag[3]):
            #	Calculate geometric factors associated with extinction and scattering 
            #	Initialise sums
        ks	= 0
        ko	= 0
        bf	= 0
        sob	= 0
        sof	= 0

        #	Weighted sums over LIDF
        
        ttl = litab# leaf inclination discrete values
        ctl = np.cos(rd*ttl)
        #	SAIL volume scattering phase function gives interception and portions to be 
        #	multiplied by rho and tau
        for i in range(na):
            chi_s,chi_o,frho,ftau= self._volscatt(tts,tto,psi,ttl[i])

        #********************************************************************************
        #*                   SUITS SYSTEM COEFFICIENTS 
        #*
        #*	ks  : Extinction coefficient for direct solar flux
        #*	ko  : Extinction coefficient for direct observed flux
        #*	att : Attenuation coefficient for diffuse flux
        #*	sigb : Backscattering coefficient of the diffuse downward flux
        #*	sigf : Forwardscattering coefficient of the diffuse upward flux
        #*	sf  : Scattering coefficient of the direct solar flux for downward diffuse flux
        #*	sb  : Scattering coefficient of the direct solar flux for upward diffuse flux
        #*	vf   : Scattering coefficient of upward diffuse flux in the observed direction
        #*	vb   : Scattering coefficient of downward diffuse flux in the observed direction
        #*	w   : Bidirectional scattering coefficient
        #********************************************************************************

            #	Extinction coefficients
            ksli = chi_s/cts
            koli = chi_o/cto

            #	Area scattering coefficient fractions
            sobli	= frho*pi/ctscto
            sofli	= ftau*pi/ctscto
            bfli	= ctl[i]*ctl[i]
            ks	= ks+ksli*lidf[i]
            ko	= ko+koli*lidf[i]
            bf	= bf+bfli*lidf[i]
            sob	= sob+sobli*lidf[i]
            sof	= sof+sofli*lidf[i]

        #	Geometric factors to be used later with rho and tau
        sdb	= 0.5*(ks+bf)
        sdf	= 0.5*(ks-bf)
        dob	= 0.5*(ko+bf)
        dof	= 0.5*(ko-bf)
        ddb	= 0.5*(1.+bf)
        ddf	= 0.5*(1.-bf)
        
        #if(flag[4]):
        #	Here rho and tau come in
        sigb= ddb*rho+ddf*tau
        sigf= ddf*rho+ddb*tau
        att	= 1.-sigf
        m2=(att+sigb)*(att-sigb)
        m2[np.where(m2<0)]=0
        m=np.sqrt(m2)
        sb	= sdb*rho+sdf*tau
        sf	= sdf*rho+sdb*tau
        vb	= dob*rho+dof*tau
        vf	= dof*rho+dob*tau
        w	= sob*rho+sof*tau
        
        #if (flag[5]):
        #	Here the LAI comes in
        #   Outputs for the case LAI = 0
        if (lai<=0):
            tss		= 1.
            too		= 1.
            tsstoo	= 1.
            rdd		= 0.
            tdd		= 1.
            rsd		= 0.
            tsd		= 0.
            rdo		= 0.
            tdo		= 0.
            rso		= 0.
            rsos	= 0.
            rsod	= 0.

            rddt	= rsoil
            rsdt	= rsoil
            rdot	= rsoil
            rsodt	= 0.
            rsost	= rsoil
            rsot	= rsoil
        
        else:
            #	Other cases (LAI > 0)
            e1		= np.exp(-m*lai)
            e2		= e1*e1
            rinf	= (att-m)/sigb
            rinf2	= rinf*rinf
            re		= rinf*e1
            denom	= 1.-rinf2*e2
        
            J1ks= self._Jfunc1(ks,m,lai)
            J2ks= self._Jfunc2(ks,m,lai)
            J1ko= self._Jfunc1(ko,m,lai)
            J2ko= self._Jfunc2(ko,m,lai)
        
            Ps = (sf+sb*rinf)*J1ks
            Qs = (sf*rinf+sb)*J2ks
            Pv = (vf+vb*rinf)*J1ko
            Qv = (vf*rinf+vb)*J2ko
        
            rdd	= rinf*(1.-e2)/denom
            tdd	= (1.-rinf2)*e1/denom
            tsd	= (Ps-re*Qs)/denom
            rsd	= (Qs-re*Ps)/denom
            tdo	= (Pv-re*Qv)/denom
            rdo	= (Qv-re*Pv)/denom
        
            tss	= np.exp(-ks*lai)
            too	= np.exp(-ko*lai)
            z	= self._Jfunc3(ks,ko,lai)
            g1	= (z-J1ks*too)/(ko+m)
            g2	= (z-J1ko*tss)/(ks+m)
        
            Tv1 = (vf*rinf+vb)*g1
            Tv2 = (vf+vb*rinf)*g2
            T1	= Tv1*(sf+sb*rinf)
            T2	= Tv2*(sf*rinf+sb)
            T3	= (rdo*Qs+tdo*Ps)*rinf
        
            #	Multiple scattering contribution to bidirectional canopy reflectance
            rsod = (T1+T2-T3)/(1.-rinf2)
            
            #if (flag[6]):
            #	Treatment of the hotspot-effect
            alf=1e6
            #	Apply correction 2/(K+k) suggested by F.-M. Bréon
            if (q>0.):
                alf=(dso/q)*2./(ks+ko)
            if (alf>200.):	#inserted H. Bach 1/3/04
                alf=200.
            if (alf==0.):
                #	The pure hotspot - no shadow
                tsstoo = tss
                sumint = (1-tss)/(ks*lai)
            else:
                #	Outside the hotspot
                fhot=lai*np.sqrt(ko*ks)
                #	Integrate by exponential Simpson method in 20 steps
                #	the steps are arranged according to equal partitioning
                #	of the slope of the joint probability function
                x1=0.
                y1=0.
                f1=1.
                fint=(1.-np.exp(-alf))*.05
                sumint=0.
        
                for i in range(20):
                    if (i<19):
                        x2=-np.log(1.-(i+1)*fint)/alf
                    else:
                        x2=1.
                    y2=-(ko+ks)*lai*x2+fhot*(1.-np.exp(-alf*x2))/alf 
                    f2=np.exp(y2)
                    sumint=sumint+(f2-f1)*(x2-x1)/(y2-y1)
                    x1=x2
                    y1=y2
                    f1=f2
                tsstoo=f1
        
        #	Bidirectional reflectance
        #	Single scattering contribution
            rsos = w*lai*sumint
            
            #	Total canopy contribution
            rso=rsos+rsod
            
            #	Interaction with the soil
            dn=1.-rsoil*rdd
            
            # rddt: bi-hemispherical reflectance factor
            rddt=rdd+tdd*rsoil*tdd/dn
            # rsdt: directional-hemispherical reflectance factor for solar incident flux
            rsdt=rsd+(tsd+tss)*rsoil*tdd/dn
            # rdot: hemispherical-directional reflectance factor in viewing direction    
            rdot=rdo+tdd*rsoil*(tdo+too)/dn
            # rsot: bi-directional reflectance factor
            rsodt=rsod+((tss+tsd)*tdo+(tsd+tss*rsoil*rdd)*too)*rsoil/dn
            rsost=rsos+tsstoo*rsoil
            rsot=rsost+rsodt
        
        return rsot, rdot, rsdt, rddt

    ## File I/O

    def _dataSpec_P5B(self): # I've reorganized the data so it's separated by row in a CSV file - much easier for handling in Python.
        try:
            infile=open(os.path.abspath(os.curdir)+'/dataSpec_P5.csv', 'r')
            print('Successfully opened dataSpec_P5.csv, reading data.')
        except:
            print('Cannot open dataSpec_P5.csv, exiting.')
            headers, data= False, False
            return headers, data
        lines=infile.readlines()
        headers=[] # Row headers, describe what these data sets mean.  Not sure they're of much utility in the code.
        data=[]
        for line in lines:
            line=line.rstrip()
            vals=line.split(',')
            headers.append(vals[0])
            datavals=vals[1:]
            #if vals[0] == 'Wavelength (nm)':
            #    datavals=[int(v) for v in datavals]
            #else:
            datavals=[float(v) for v in datavals]
            data.append(datavals)
        infile.close()
        return headers, data

    def _writeoutput(self, l, resh, resv, outfile):
        if len(outfile)==0:
            outfile='Refl_CAN.txt'
        if outfile.endswith('.csv'):
            delimiter=','
        else:
            delimiter='\t'
        output=open(outfile,'w')
        output.write('Wavelength (nm)'+delimiter+'Hemispherical reflectance'+delimiter+'Directional reflectance\n')
        for i in range(len(l)):
            output.write(str(l[i])+delimiter+str(resh[i])+delimiter+str(resv[i])+'\n')
        output.close()

    #****************************************************
    def _writeconfig(self, paramnames, paramlist, outname):
        output=open(outname,'w')
        for i in range(len(paramnames)):
            output.write(paramnames[i]+','+str(paramlist[i])+'\n')
        output.close()

    ## J functions

    def _Jfunc1(self, k,l,t):
    #	J1 function with avoidance of singularity problem
    #	
        d=(k-l)*t
        Jout=np.zeros(d.size)
        for i in range(Jout.size):
            if (np.absolute(d[i])>1e-3):
                Jout[i]=(np.exp(-l[i]*t)-np.exp(-k*t))/(k-l[i])
            else:
                Jout[i]=0.5*t*(np.exp(-k*t)+np.exp(-l[i]*t))*(1.-d[i]*d[i]/12.)
        return Jout

    def _Jfunc2(self, k,l,t):
        Jout=(1.-np.exp(-(k+l)*t))/(k+l)
        return Jout

    def _Jfunc3(self, k,l,t):
    #	J2 function
        Jout=(1.-np.exp(-(k+l)*t))/(k+l)
        return Jout

    ## Main section

    def _canref(self, rsot, rdot, rsdt, rddt, Es, Ed, tts):
    #    direct / diffuse light	#
    #
    # the direct and diffuse light are taken into account as proposed by:
    # Francois et al. (2002) Conversion of 400–1100 nm vegetation albedo 
    # measurements into total shortwave broadband albedo using a canopy 
    # radiative transfer model, Agronomie
        skyl	=	0.847- 1.61*np.sin(np.radians(90-tts))+ 1.04*np.sin(np.radians(90-tts))*np.sin(np.radians(90-tts)) # % diffuse radiation
    # Es = direct
    # Ed = diffuse
    # PAR direct
        PARdiro	=	(1-skyl)*Es
    # PAR diffus
        PARdifo	=	(skyl)*Ed
    # resh : hemispherical reflectance
        resh  = (rddt*PARdifo+rsdt*PARdiro)/(PARdiro+PARdifo)
    # resv : directional reflectance
        resv	= (rdot*PARdifo+rsot*PARdiro)/(PARdiro+PARdifo)
        return resh, resv

    # Common leaf distributions
    Planophile = (1, 0)
    Erectophile = (-1, 0)
    Plagiophile = (0, -1)
    Extremophile = (0, 1)

    def run(self, N, Cab, Car, Cbrown, Cw, Cm, psoil, LAI, hspot, tts, tto, psi, LIDF, outname=None, Py6S=False):
        # Deal with the LIDF 
        try:
            l = len(LIDF)
            if l != 2:
                # Raise error
                pass

            TypeLidf = 1
            LIDFa = LIDF[0]
            LIDFb = LIDF[1]
        except TypeError:
            TypeLidf = 2
            LIDFa = LIDF
            LIDFb = 0

        # LIDF output
        lidf=self._calcLidf(TypeLidf,LIDFa,LIDFb)

        # Spectra data import
        headers,spectra=self._dataSpec_P5B()    

        # PROSPECT output
        #LEAF CHEM & STR PROPERTIES#
        rho, tau= self._prospect_5B(N,Cab,Car,Cbrown,Cw,Cm,spectra)

        #
        #   Soil Reflectance Properties #
        #
        # rsoil1 = dry soil
        # rsoil2 = wet soil
        Rsoil1=np.array(spectra[9])#
        Rsoil2=np.array(spectra[10])#
        rsoil0=self._soilref(psoil,Rsoil1,Rsoil2)


        #
        #        CALL PRO4SAIL         #
        #
        rsot, rdot, rsdt, rddt= self._PRO4SAIL(rho, tau,lidf,LAI,hspot,tts,tto,psi,rsoil0)
        Es=np.array(spectra[7])#
        Ed=np.array(spectra[8])#
        #
        #   
        resh, resv = self._canref(rsot, rdot, rsdt, rddt, Es, Ed, tts)

        if outname is not None:
            # Writing output to disk
            self._writeoutput(spectra[0],resh,resv,outname)

        spectra[0] = np.array(spectra[0])

        if Py6S:
            arr = np.transpose(np.vstack( (spectra[0]/1000.0, resh) ))
            return arr
        else:
            return (spectra[0], resh, resv)

    def _main_PROSAIL(self):
        """
        This program allows modeling reflectance data from canopy
        - modeling leaf optical properties with PROSPECT-5 (feret et al. 2008)
        - modeling leaf inclination distribution function with the subroutine campb
        (Ellipsoidal distribution function caracterised by the average leaf 
        inclination angle in degree), or dladgen (2 parameters LIDF)
        - modeling canopy reflectance with 4SAIL (Verhoef et al., 2007)
        This version has been implemented by Jean-Baptiste Feret
        Jean-Baptiste Feret takes the entire responsibility for this version 
        All comments, changes or questions should be sent to:
        jbferet@stanford.edu

        References:
            Verhoef et al. (2007) Unified Optical-Thermal Four-Stream Radiative
            Transfer Theory for Homogeneous Vegetation Canopies, IEEE TRANSACTIONS 
            ON GEOSCIENCE AND REMOTE SENSING, VOL. 45, NO. 6, JUNE 2007
            Féret et al. (2008), PROSPECT-4 and 5: Advances in the Leaf Optical
            Properties Model Separating Photosynthetic Pigments, REMOTE SENSING OF 
            ENVIRONMENT
        The specific absorption coefficient corresponding to brown pigment is
        provided by Frederic Baret (EMMAH, INRA Avignon, baret@avignon.inra.fr)
        and used with his autorization.
        the model PRO4SAIL is based on a version provided by
            Wout Verhoef
            NLR
            April/May 2003

         The original 2-parameter LIDF model is developed by and described in:
            W. Verhoef, 1998, "Theory of radiative transfer models applied in
            optical remote sensing of vegetation canopies", Wageningen Agricultural
            University, The Netherlands, 310 pp. (Ph. D. thesis)
         the Ellipsoidal LIDF is taken from:
           Campbell (1990), Derivtion of an angle density function for canopies 
           with ellipsoidal leaf angle distribution, Agricultural and Forest 
           Meteorology, 49 173-176
        """

    # Spectra data import
        headers,spectra = self._dataSpec_P5B()

    # LIDF output

        TypeLidf=1
        LIDFa	=	-0.35
        LIDFb	=	-0.15
    # if 2-parameters LIDF: TypeLidf=1
    #     if (TypeLidf==1):
    #     # LIDFa LIDF parameter a, which controls the average leaf slope
    #     # LIDFb LIDF parameter b, which controls the distribution's bimodality
    #         #	LIDF type 		a 		 b
    #         #	Planophile 		1		 0
    #         #	Erectophile    -1	 	 0
    #         #	Plagiophile 	0		-1
    #         #	Extremophile 	0		 1
    #         #	Spherical 	   -0.35 	-0.15
    #         #	Uniform 0 0
    #     # 	requirement: |LIDFa| + |LIDFb| < 1	
    #         LIDFa	=	-0.35
    #         LIDFb	=	-0.15
    # # if ellipsoidal distribution: TypeLidf=2
    #     elif (TypeLidf==2):
    #     # 	LIDFa	= average leaf angle (degrees) 0 = planophile	/	90 = erectophile
    #     # 	LIDFb = 0
    #         LIDFa	=	30
    #         LIDFb	=	0
    #     #	Generate leaf angle distribution from average leaf angle (ellipsoidal) or (a,b) parameters
    #     if(TypeLidf==1):
    #         lidf= dladgen(LIDFa,LIDFb)
    #     elif(TypeLidf==2):
    #         lidf=calc_LIDF_ellipsoidal(na,LIDFa)
        lidf = self._calcLidf(TypeLidf,LIDFa,LIDFb)

    # PROSPECT output
    #LEAF CHEM & STR PROPERTIES#
    #
    # INITIAL PARAMETERS
        Cab		=	40.	# chlorophyll content (µg.cm-2) 
        Car		=	8.	# carotenoid content (µg.cm-2)
        Cbrown	=	0.0	# brown pigment content (arbitrary units)
        Cw		=	0.01# EWT (cm)
        Cm		=	0.009# LMA (g.cm-2)
        N		=	1.5	# structure coefficient
        rho, tau= self._prospect_5B(N,Cab,Car,Cbrown,Cw,Cm,spectra)

    #
    #	Soil Reflectance Properties	#
    #
    # rsoil1 = dry soil
    # rsoil2 = wet soil
        psoil	=	1.	# soil factor (psoil=0: wet soil / psoil=1: dry soil)
        Rsoil1=np.array(spectra[9])#
        Rsoil2=np.array(spectra[10])#
        rsoil0= self._soilref(psoil,Rsoil1,Rsoil2)

    #
    #	4SAIL canopy structure parm	#
    #
        LAI		=	3.	# leaf area index (m^2/m^2)
        hspot	=	0.01# hot spot
        tts		=	30.	# solar zenith angle (°)
        tto		=	10.	# observer zenith angle (°)
        psi		=	0.	# azimuth (°)

    #
    #        CALL PRO4SAIL         #
    #
        rsot, rdot, rsdt, rddt= self._PRO4SAIL(rho, tau,lidf,LAI,hspot,tts,tto,psi,rsoil0)
        Es=np.array(spectra[7])#
        #print('Es is length: '+str(Es.size))
        Ed=np.array(spectra[8])#
    #
    #	
        resh, resv= self._canref(rsot, rdot, rsdt, rddt, Es, Ed, tts)
        # Writing output to disk
        outname='Refl_CAN_P5B.txt'
        self._writeoutput(spectra[0],resh,resv,outname)
        outname='prosail.cfg'
        #	Before returning, save current parameters as old ones
        paramnames=['Structure coefficient N', 
            'Chlorophyll content (µg.cm-2) Cab', 
            'Carotenoid content (µg.cm-2) Car', 
            'Brown pigment content (arbitrary units) Cbrown', 
            'Equivalent water thickness (cm) Cw', 
            'LIDFa',
            'LIDFb',
            'LIDF Type TypeLidf',
            'Leaf mass per unit leaf area (g.cm-2) Cm', 
            'Leaf area index LAI', 
            'Hot spot hspot', 
            'Solar zenith angle (°) tts', 
            'Observer zenith angle (°) tto', 
            'Azimuth (°) psi',
            'Soil coefficient psoil']
        
        paramlist=[N,Cab,Car,Cbrown,Cw,Cm,LIDFa,LIDFb,TypeLidf,LAI,hspot,tts,tto,psi,psoil]
        self._writeconfig(paramnames, paramlist, outname)
