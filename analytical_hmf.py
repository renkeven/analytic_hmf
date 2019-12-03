import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from scipy import interpolate
from matplotlib import rc
import seaborn as sns

m_c = sns.color_palette("Paired",12).as_hex()
m_c[10] = '#e0a061'

rc('font', **{'family': 'serif'})
rc('text', usetex=True)

z = 0.0

#--- PLANCK used in TIAMAT

#o_m = 0.308
#o_l = 1. - o_m
#H_0 = 67.8

#---PLANCK GENESIS
o_m = 0.3121
o_l = 1. - o_m
H_0 = 67.51
h = H_0/100.

#--- WMAP

#o_m = 0.272
#o_l = 1. - o_m
#H_0 = 70.4

G = 6.67e-11*1e-6*3.2408e-23*1.99e30                    #km^2 Mpc/M_sol/s^2
const = (3*(H_0**2))/(8*np.pi*G)

mdens = o_m*const
#*(o_m*((1+z)**3.)+o_l)

### Growth FACTOR

def invbeta(z):
        bete = ((1-o_m)*((1+z)**-3.))/(o_m + (1-o_m)*((1+z)**-3.))
        return (5./6.) * scipy.special.betainc((5./6.),(2./3.),bete) * scipy.special.beta((5./6.),(2./3.)) * (o_m/(1-o_m))**(1./3.) * np.sqrt(1+ (o_m/((1-o_m)*((1+z)**-3.))))

def growth(z):
        return invbeta(z)/invbeta(0)

## DATA

power_spectrum = np.genfromtxt('cosmology/genesis_powerspec.txt', comments='#')
power_spectrum[:,0] = power_spectrum[:,0]*(H_0/100.)
power_spectrum[:,1] = (growth(z)**2.)*power_spectrum[:,1]/((H_0/100.)**3)


##Functions to HMF

def growth2(z):
	zint = lambda z: (1+z)*np.power((o_m*(1+z)**3 + o_l),(-3./2.))
	return 2.5 * o_m * np.sqrt((o_m*(1+z)**3 + o_l)) * scipy.integrate.quad(zint, z, np.inf)[0]


def r_to_m(r):
	return (4*np.pi*mdens/3.)*(r**3.)

def m_to_r(m):
	return np.power(((3.*m)/(4*np.pi*mdens)),(1./3.))

def w2(m):
	kr = power_spectrum[:,0] * m_to_r(m)
	return (3.*(np.sin(kr)-(kr*np.cos(kr)))/(kr**3.))**2.

def sigma(m):
	c = 1./(2*(np.pi**2.))
	o2 = (power_spectrum[:,0]**2.) * power_spectrum[:,1] * w2(m)
	return np.sqrt(c * scipy.integrate.simps(o2,power_spectrum[:,0]))

def dw2dm(m):
	kr = power_spectrum[:,0] * m_to_r(m)
	return (np.sin(kr) - kr*np.cos(kr)) * (np.sin(kr)*(1. - 3./(kr**2)) + 3*np.cos(kr)/kr)

def dlnodlnm(m):
	r = m_to_r(m)
	a = 3./(2.*(np.pi**2.)*(sigma(m)**2.)*(r**4.))
	i = dw2dm(m)*power_spectrum[:,1]/(power_spectrum[:,0]**2.)

	return np.abs(a * scipy.integrate.simps(i,power_spectrum[:,0]))

## Fits
def ps(m):
        return np.sqrt(2./np.pi) * (1.69/sigma(m)) * np.exp(-(1.69**2.)/(2.*(sigma(m)**2.)))

def watson(a):
        return 0.282*((1.406/sigma(a))**(2.163) + 1.) * np.exp(-1.210/(sigma(a)**2))

def greg(a):
        f = 0.0333* ((12.33/sigma(a)) + 1.)**(1.153) * np.exp(-1.01/(sigma(a)**2))
        return f

def elahi(a):
        f = (10**17.22 * 1.199e-19)  * ((4.42 /sigma(a)) + 1.)**(0.60)  * np.exp(-1.23/(sigma(a)**2))
        return f

def elahi_new(a):
        f = (10**-1.97) * ((9.74 /sigma(a)) + 1.)**(1.34)  * np.exp(-1.51/(sigma(a)**2))
        return f    

def jenkin(a):
        return 0.315*np.exp(-np.abs(-np.log(sigma(a)) + 0.61)**3.8)

def st(a):
	A = 0.3222
	b = 0.707
	v = 1.69/sigma(a)
	return A * np.sqrt(2.*b/np.pi) * (1. + (1./(b*(v**2.)))**(0.3)) * v * np.exp(-b/2. * (v**2.))

def tinker(a):

        v = 1.69/sigma(a)
        a0 = 0.368
        b0 = 0.589*((1+z)**(0.2))
        p0 = -0.729*((1+z)**(-0.8))
        n0 = -0.243*((1+z)**(0.27))
        g0 = 0.864*((1+z)**(-0.01))

        return a0 * (1+(b0*v)**(-2*p0)) * v**(2*n0) * np.exp(-g0 * (v**2.) /2.)	

##10^6 - 10^16 halos spaced logarithmically
## outputs are units of dn/dlog10m

xspace = np.logspace(6,16,5000)

ytinker = np.zeros(len(xspace))
ywatson = np.zeros(len(xspace))
ygreg = np.zeros(len(xspace))
yelahi = np.zeros(len(xspace))
yps = np.zeros(len(xspace))
yst = np.zeros(len(xspace))
ysigma = np.zeros(len(xspace))
yelahi_new = np.zeros(len(xspace))
yjenkin = np.zeros(len(xspace))

for i in range(len(xspace)):
    yjenkin[i] = mdens/xspace[i] * jenkin(xspace[i]) * dlnodlnm(xspace[i])/np.log10(np.exp(1))
    ytinker[i] = mdens/xspace[i] * tinker(xspace[i]) * dlnodlnm(xspace[i])/np.log10(np.exp(1))
    ywatson[i] = mdens/xspace[i] * watson(xspace[i]) * dlnodlnm(xspace[i])/np.log10(np.exp(1))
    yelahi[i] = mdens/xspace[i] * elahi(xspace[i]) * dlnodlnm(xspace[i])/np.log10(np.exp(1))
    ygreg[i] = mdens/xspace[i] * greg(xspace[i]) * dlnodlnm(xspace[i])/np.log10(np.exp(1))
    yps[i] = mdens/xspace[i] * ps(xspace[i]) * dlnodlnm(xspace[i])/np.log10(np.exp(1))
    yst[i] = mdens/xspace[i] * st(xspace[i]) * dlnodlnm(xspace[i])/np.log10(np.exp(1))
    yelahi_new[i] = mdens/xspace[i] * elahi_new(xspace[i]) * dlnodlnm(xspace[i])/np.log10(np.exp(1))
    ysigma[i] = sigma(xspace[i])


#load example hmf
example = np.genfromtxt('examples/mVector_PLANCK-SMT .txt',comments='#')

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plot(xspace,yjenkin, label='jenkin fit')
ax1.plot(xspace,ygreg, label='poole fit')
ax1.plot(xspace,yelahi, label='elahi fit (old)')
ax1.plot(xspace,yelahi_new, label='elahi fit (new)')


ax1.plot(xspace,yst, label='analytical sheth-tormen', c='#000000')
ax1.scatter(example[:,0][::2]/h, example[:,7][::2]*(h**3), s=8, marker='+', label='z=0 hmfcalc check sheth-tormen', c='#000000')



ax1.set_xscale('log', basex=10)
ax1.set_yscale('log', basey=10)
ax1.set_xlim(10**7, 10**15)
ax1.set_ylim(10**-8, 10**3)
ax1.set_xlabel('$M_{h}$ [$M_{\odot}$]')
ax1.set_ylabel('$dN/d\log M$ [Mpc$^-3$]')

ax1.legend(framealpha=0.)

plt.savefig('hmfcheck.png')



