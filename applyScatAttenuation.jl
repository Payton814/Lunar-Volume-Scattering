using Plots
using CSV
using FFTW
using DataFrames
using Interpolations

function applyScatAttenuation(t, waveform, scat_model::String, d)



## Read in the Scattering powering attenuation
dfmatlab = CSV.read("./VolumeScatModels/" * scat_model, DataFrame)
## Data is read in as dB/m, need to convert to linear scale. Since its in Power we divide by 20 to apply to E-field
S21 = 10 .^(d .* dfmatlab[:, 1] ./20)
## It is assumed for now that the attenuation model is valid from 0 to 1.2 GHz, in steps of 0.001
fmatlab = range(start = 0, stop = 1.2, step = 0.001)


## FFT the input waveform and grab the frequency spectrum
fE = fft(waveform)
freq = fftfreq(length(waveform), 1/(t[2] - t[1]))

## Since the scattering attenuation is not the same number of points as the input waveform
## nor cover the same frequency range, so we need to find were 1.2 and -1.2 are in the input
## waveform frequency spectrum.
fend = argmin(abs.(1.2 .- freq))
fend2 = argmin(abs.(-1.2 .- freq))

## Need to build an interpolation on the scattering attenuation so that we can resample it to fft_Efield
## to the input waveform.
Pmat_interp = linear_interpolation(fmatlab, S21)

## Make an array of ones that will be applied to input waveform. If left as all 1 then no attenuation is applied
Pmat = ones(length(freq))

## The relevant frequencies need to be given our scattering attenuation
## The reason we do positive and negative is because a FFT must be hermitian
## but we only have an absolute power attenuation. This is equivalent to saying
## there is real power loss with no phase shifting
Pmat[1:fend] = Pmat_interp.(freq[1:fend])

## The -1 here is so the arrays are the right length. I didnt put much thought into this
## so this could be an off by 1 error and should be looked into later.
Pmat[fend2:end] = reverse(Pmat_interp.(freq[1:fend-1]))

## apply the attenuation to the fft of the input waveform
## then perform ifft to revert back to time-domain
## take the real part since thats all that matters
ifE = real(ifft(fE .* Pmat))
println(length(ifE))

return ifE

#scatter(freq, Pmat)
#savefig("matlab_resample.pdf")

#plot(freq, log.(10, abs.(fE)))
#savefig("fft_Efield.pdf")

#plot(t, Ez, label = "Input waveform")
#plot!(t, ifE, label = "Attenuated")
#savefig("Efield.pdf")

end