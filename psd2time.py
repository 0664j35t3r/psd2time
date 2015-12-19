#!/usr/bin/python
import numpy
import scipy as sp
import scipy.linalg
import scipy.constants
import scipy.fftpack
import math
import matplotlib
import matplotlib.pyplot
import matplotlib.mlab
import sys

if len(sys.argv) != 3:
	print "Create time series from (cross-)PSDs"
	print ""
	print ""
	print "Usage: ./psd2time.py [parameter file] [output file]"
	print ""
	#print "The parameter file must have the following syntax:"
	#print ""
	#print ""
	
	exit(0)





param_file_name = sys.argv[1]    #parameter file

output_file_series = sys.argv[2] #output file

sampling_rate = -1.
num_points = -1
interpolation_kind = -1
input_type = -1
output_type = -1
psd_file_names = {}
plot_series_data = []
plot_psd_data = []

#parse parameter file
f = open(param_file_name, 'r')
for line in f:
	if line.strip()=="":
		continue
	if line.strip()[0]=="#":
		continue
	a = line.strip().split(" ")
	if len(a)==2 and a[0]=="sampling_rate":
		sampling_rate = float(a[1])
	if len(a)==2 and a[0]=="num_points":
		num_points = int(a[1])
	if len(a)==2 and a[0]=="interpolation":
		if a[1]=="linear":
			interpolation_kind = 0
		if a[1]=="logarithmic":
			interpolation_kind = 1
	if len(a)==2 and a[0]=="input":
		if a[1]=="acceleration":
			input_type = 0
		if a[1]=="velocity":
			input_type = 1
		if a[1]=="displacement":
			input_type = 2
	if len(a)==2 and a[0]=="output":
		if a[1]=="acceleration":
			output_type = 0
		if a[1]=="velocity":
			output_type = 1
		if a[1]=="displacement":
			output_type = 2
	if len(a)==4 and a[0]=="psd":
		row = int(a[1])
		col = int(a[2])
		psd_file_names[(row, col)] = a[3]
	if len(a)==5 and a[0]=="plot" and a[1]=="series":
		plot_series_data.append([int(a[2]), int(a[3]), a[4]])
	if len(a)==5 and a[0]=="plot" and a[1]=="psd":
		plot_psd_data.append([int(a[2]), int(a[3]), a[4]])
		
f.close()

#test parameter 
if sampling_rate < 0:
	exit(0)
if num_points < 0:
	exit(0)
if interpolation_kind < 0:
	exit(0)
if input_type < 0:
	exit(0)
if output_type < 0:
	exit(0)

def readPsd(file_name):
	f = open(file_name, 'r')
	xVals = []
	yVals = []
	for line in f:
		if line.strip()[0]=="#":
			continue
		a = line.split(" ")
		assert len(a)>=2
		xVals.append(float(a[0]))
		if len(a)==2:
			yVals.append(complex(float(a[1]),0))
		else:
			yVals.append(complex(float(a[1]),float(a[2])))
	f.close()
	return numpy.array([xVals, yVals])


def calc_exp_Z_minus_1_div_Z(jZ):
	if numpy.absolute(jZ)<1e-5:
		return 1.+(1.+(1.+jZ/4.)*jZ/3.)*jZ/2.
	else:
		return (numpy.exp(jZ)-1.)/jZ

def integratePowerFunction(power, limit_low, limit_high):
	log_limit_high = numpy.log(limit_high)
	log_limit_low = numpy.log(limit_low)
	#return log_limit_high*calc_exp_Z_minus_1_div_Z(log_limit_high*(power+1))-log_limit_low*calc_exp_Z_minus_1_div_Z(log_limit_low*(power+1))
	log_limit_diff = log_limit_high - log_limit_low
	assert not numpy.math.isnan(log_limit_diff)
	assert not numpy.math.isnan(numpy.exp(log_limit_high*(power+1)))
	assert not numpy.math.isnan(calc_exp_Z_minus_1_div_Z(-(power+1)*log_limit_diff))
	result = numpy.exp(log_limit_high*(power+1))*(log_limit_diff*calc_exp_Z_minus_1_div_Z(-(power+1)*log_limit_diff))
	if numpy.math.isnan(result):
		print "integratePowerFunction", power, limit_low, limit_high
	return result

def integrateSplineLin(x_low, x_high, y_low, y_high, limit_low, limit_high, order):
	assert limit_high <= x_high
	assert limit_low >= x_low
	
	if limit_high<=limit_low:
		return 0.
	else:
		A = (y_high - y_low)/(x_high - x_low); #order+1
		B = (y_low*x_high - y_high*x_low)/(x_high - x_low); #order
		
		return A*integratePowerFunction(order+1, limit_low, limit_high)+B*integratePowerFunction(order, limit_low, limit_high)

#todo phase correction!!!
def integrateSplineLogOld(x_low, x_high, y_low, y_high, limit_low, limit_high, order):
	if limit_high<=limit_low:
		return 0.
	else:
		if y_low == 0 or y_high == 0:
			return 0.
		log_y_low = numpy.log(y_low)
		log_y_high = numpy.log(y_high)
		
		if log_y_high.imag - log_y_low.imag > numpy.pi:
			log_y_low.imag -= 2*numpy.pi
		elif log_y_high.imag - log_y_low.imag < -numpy.pi:
			log_y_low.imag += 2*numpy.pi
		
		A = (numpy.log(x_high)*log_y_low-numpy.log(x_low)*log_y_high)/(numpy.log(x_high)-numpy.log(x_low))
		assert not numpy.math.isnan(A)
		B = (log_y_high-log_y_low)/(numpy.log(x_high)-numpy.log(x_low)) + order
		assert not numpy.math.isnan(A)
		
		return numpy.exp(A)*integratePowerFunction(B, limit_low, limit_high)
		
#todo phase correction!!!
def integrateSplineLog(x_low, x_high, y_low, y_high, limit_low, limit_high, order):
	if limit_high<=limit_low:
		return 0.
	else:
		if y_low == 0 or y_high == 0:
			return 0.
		log_y_low = numpy.log(y_low)
		log_y_high = numpy.log(y_high)
		
		if log_y_high.imag - log_y_low.imag > numpy.pi:
			log_y_low.imag -= 2*numpy.pi
		elif log_y_high.imag - log_y_low.imag < -numpy.pi:
			log_y_low.imag += 2*numpy.pi
		
		A = (numpy.log(x_high)*log_y_low-numpy.log(x_low)*log_y_high)/(numpy.log(x_high)-numpy.log(x_low))
		assert not numpy.math.isnan(A)
		B = (log_y_high-log_y_low)/(numpy.log(x_high)-numpy.log(x_low)) + order +1 
		assert not numpy.math.isnan(A)
		
		#return numpy.exp(A)*integratePowerFunction(B, limit_low, limit_high)
		
		log_limit_low = numpy.log(limit_low)
		log_limit_high = numpy.log(limit_high)
		log_limit_diff = log_limit_high-log_limit_low
		
		return numpy.exp(A+B*log_limit_high)*log_limit_diff*calc_exp_Z_minus_1_div_Z(-B*log_limit_diff)
		

def integrateSplineLogTest():
	x_low = 1.5
	x_high = 3
	y_low = 12
	y_high=89
	limit_low = 2.3
	limit_high = 3
	order = 4
	print integrateSplineLog(x_low, x_high, y_low, y_high, limit_low, limit_high, order),integrateSplineLogOld(x_low, x_high, y_low, y_high, limit_low, limit_high, order)
	
	
#integrateSplineLogTest()
#exit(0)


def integrateSpline(x_low, x_high, y_low, y_high, limit_low, limit_high, kind, order):
	result = 0
	if kind==0:
		result = integrateSplineLin(x_low, x_high, y_low, y_high, limit_low, limit_high, order)
	else:
		result = integrateSplineLog(x_low, x_high, y_low, y_high, limit_low, limit_high, order)
	if numpy.math.isnan(result):
		print x_low
		print x_high
		print y_low
		print y_high
		print limit_low
		print limit_high
		print order
	assert not numpy.math.isnan(result)
	return result

def mapPsdToSpecLines(num_points, sampling_rate, psd, interpolation_kind, order):
	
	delta_freq = sampling_rate/num_points/2.
	result = numpy.zeros((1, num_points),dtype=complex)
	frequency_limits = numpy.linspace(0,num_points, num_points+1)*delta_freq

#    print "frequencies = " , frequency_limits

	for i in range(0, psd.shape[1]-1):
		x_low = psd[0][i]
		x_high = psd[0][i+1]
		y_low = psd[1][i]
		y_high = psd[1][i+1]
		
		idx_min=max(int(math.floor(x_low/delta_freq)),0)
		idx_max=min(int(math.ceil(x_high/delta_freq)), num_points)

		for idx_low in range(idx_min, idx_max):
			idx_high = idx_low + 1
			result[0,idx_low] += integrateSpline(x_low, x_high, y_low, y_high, max(frequency_limits[idx_low], x_low), min(frequency_limits[idx_high], x_high), interpolation_kind, 2*order)

	for i in range(1,(int(num_points)+1)/2):
		tmp = (result[0, 2*i-1]+ result[0, 2*i])*0.5
		result[0, 2*i-1] = tmp
		result[0, 2*i] = tmp
		
		
	result*=numpy.power(2.*numpy.pi, 2*order)

	return result

def sqrtMatrix(matrix):
	
	w,v = sp.linalg.eigh(matrix)
	assert v.min >=0

	c = numpy.dot(numpy.dot(v,sp.diag(sp.sqrt(w),0)),v.conjugate().transpose())

	assert sp.linalg.norm(numpy.real(numpy.dot(c, c.transpose().conjugate())-matrix))<= 1e-15 * sp.linalg.norm(numpy.real(matrix))
	assert sp.linalg.norm(numpy.imag(numpy.dot(c, c.transpose().conjugate())-matrix))<= 1e-15 * sp.linalg.norm(numpy.imag(matrix))
	
	return c

def createTimeSeries(num_points, psd_values):
	
	psd_dim = psd_values.shape[1]
	
	time_sig = numpy.random.randn(num_points, psd_dim)

	freq_sig = sp.fftpack.rfft(time_sig, axis=0)
	
	freq_sig[0,:]=numpy.dot(sqrtMatrix(psd_values[0,:,:]),freq_sig[0,:])

	for i in range(1,(int(num_points)+1)/2):
		
		complex_tmp = numpy.zeros(psd_dim,dtype=complex)
		complex_tmp.real = freq_sig[2*i-1,:]
		complex_tmp.imag = freq_sig[2*i,:]
		
		complex_tmp=numpy.dot(sqrtMatrix(psd_values[2*i-1,:,:]+psd_values[2*i,:,:]),complex_tmp)

		freq_sig[2*i-1,:] = complex_tmp.real
		freq_sig[2*i,:  ] = complex_tmp.imag

	if num_points%2 == 0:
		freq_sig[num_points-1,:]*=numpy.dot(sqrtMatrix(psd_values[num_points-1,:,:]),freq_sig[num_points-1,:])
		
	freq_sig*=sp.sqrt(num_points/2.)

	time_sig = sp.fftpack.irfft(freq_sig, axis=0)

	return time_sig

num_points_pow_2 = numpy.power(2, int(numpy.ceil(math.log(num_points,2))))
print "num_points = ", num_points
print "num_points_pow_2 = ", num_points_pow_2

psd_dim = 0
for psd_key in psd_file_names:
	psd_dim = max(psd_key[0], psd_key[1], psd_dim)
	
print "number of time series =", psd_dim


psd_values = numpy.zeros((num_points_pow_2, psd_dim, psd_dim),dtype=complex )

for psd_key in psd_file_names:
	print "processing psd", psd_key

	print "read psd from file", psd_file_names[psd_key]
	inputPSD =  readPsd(psd_file_names[psd_key])

	print "integrate psd and map to spec lines"
	psd_spec_lines = mapPsdToSpecLines(num_points_pow_2, sampling_rate, inputPSD, interpolation_kind, input_type-output_type)
	
	if psd_key[0]==psd_key[1]:
		psd_values[:,psd_key[0]-1,psd_key[1]-1] = psd_spec_lines.real
	elif psd_key[0]<psd_key[1]:
		psd_values[:,psd_key[0]-1,psd_key[1]-1] = psd_spec_lines
		psd_values[:,psd_key[1]-1,psd_key[0]-1] = psd_spec_lines.conjugate()
	elif psd_key[1]<psd_key[0]:
		psd_values[:,psd_key[0]-1,psd_key[1]-1] = psd_spec_lines.conjugate()
		psd_values[:,psd_key[1]-1,psd_key[0]-1] = psd_spec_lines

print "sum = ", numpy.sum(psd_values,axis=0)

print "create time series"
sig = createTimeSeries(num_points_pow_2, psd_values)[0:num_points]

#print "T =", num_points/sampling_rate
#print "f_delta =", delta_freq
#print "f_max =", (num_points/2.)*delta_freq
	
#print 
#print "cov(x) =", numpy.dot(sig, sig.transpose())/(num_points)
variance = numpy.var(sig, axis=0)
print "variance =", variance
print "stdev =", numpy.sqrt(variance)
print "max =", numpy.max(sig,axis=0)
print "min =", numpy.min(sig,axis=0)

print "write time series to file"
f = open(output_file_series,'w')
f.write("\n")
f.write("# time")
for k in range(0,psd_dim):
	f.write(" signal" + str(k+1))
f.write("\n")

for k in range(0, num_points):
	f.write(str(k/sampling_rate)+ ' ' +  str(sig[k,:])[1:-1].replace(","," ") + '\n')
f.close()

#plot series
for p in plot_series_data:
	print "plot series",p[0]
	fig = matplotlib.pyplot.figure() 
	fig1 = fig.add_subplot(111)
	fig1.plot(numpy.linspace(0, p[1]-1, p[1])/sampling_rate, sig[:p[1],p[0]-1])
	fig.savefig(p[2])
	
#plot psd
for p in plot_psd_data:
	print "plot cross-psd of series",p[0],"and series",p[1]
	csd = matplotlib.mlab.csd(sig[:,p[1]-1], sig[:,p[0]-1], NFFT = pow(2, 13),  Fs=sampling_rate)
	fig = matplotlib.pyplot.figure() 
	fig1 = fig.add_subplot(211)
	fig1.loglog(csd[1], numpy.absolute(csd[0]))
	fig2 = fig.add_subplot(212)
	fig2.plot(numpy.log(csd[1]), numpy.angle(csd[0]))
	fig.savefig(p[2])

exit(0)

psd11 = matplotlib.mlab.psd(sig[:,0], NFFT = pow(2, 13),  Fs=sampling_rate)
psd22 = matplotlib.mlab.psd(sig[:,1], NFFT = pow(2, 13),  Fs=sampling_rate)
psd12 = matplotlib.mlab.csd(sig[:,0], sig[:,1], NFFT = pow(2, 13),  Fs=sampling_rate)
psd21 = matplotlib.mlab.csd(sig[:,1], sig[:,0], NFFT = pow(2, 13),  Fs=sampling_rate)

#diff2 = numpy.zeros(num_points)

#diff2[1:num_points-1] = (sig[0:num_points-2]+sig[2:num_points]-2*sig[1:num_points-1])*sampling_rate*sampling_rate


#fig1 = fig.add_subplot(211)
#fig1.plot(numpy.linspace(0, 999, 1000)/sampling_rate, sig[:1000,0])
#fig2 = fig.add_subplot(212)
#fig2.plot(numpy.linspace(0, 999, 1000)/sampling_rate, sig[:1000,1])



fig11.loglog(psd11[1], psd11[0])
fig12 = fig.add_subplot(222)
fig12.loglog(psd12[1], numpy.absolute(psd12[0]))
fig21 = fig.add_subplot(223)
fig21.loglog(psd21[1], numpy.absolute(psd21[0]))
fig22 = fig.add_subplot(224)
fig22.loglog(psd22[1], psd22[0])



#fig3 = fig.add_subplot(223)
#fig3.plot(numpy.linspace(0, 999, 1000)/sampling_rate, diff2[:1000])

#psd_diff = matplotlib.mlab.psd(diff2, NFFT = pow(2, 13),  Fs=sampling_rate)
#fig4 = fig.add_subplot(111)
#fig4.loglog(psd_diff[1], psd_diff[0])

#fig4 = fig.add_subplot(224)
#fig4.loglog(inputPSD[0], inputPSD[1])
#fig2.set_xlim(fig4.get_xlim())
#fig2.set_ylim(fig4.get_ylim())
fig.savefig("dskfhsjkdhf.svg")
matplotlib.pyplot.show() 

print "finished"
