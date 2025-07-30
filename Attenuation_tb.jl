using Plots
using CSV
using DataFrames
include("applyScatAttenuation.jl")

df = CSV.read("./Noise/VolScatNoiseData/CE3_Donut_5ro_3ri_8x0y2.5z_ref.csv", DataFrame)
E = df[:, 4]
t = df[:, 1]

E_new = applyScatAttenuation(t, E, "CE3_Cuboid.csv", 2)


dftime = CSV.read("./CE3_2m_Efield.csv", DataFrame)
ttime = dftime[:, 1]
Etime = dftime[:, 2]


plot(t, E, xlimits = (45,55), label="Input waveform")
plot!(t, E_new, label="Waveguide model")
plot!(ttime, Etime, ls=:dash, label="Cylindrical Shell model")
xlabel!("Time (ns)")
ylabel!("E (V/m)")
title!("Volume Scattering attenuation at 2m")
savefig("Efield.png")