using Plots
using CSV
using DataFrames
include("applyScatAttenuation.jl")

df = CSV.read("./Noise/VolScatNoiseData/CE3_Donut_5ro_3ri_8x0y2.5z_ref.csv", DataFrame)
E = df[:, 4]
t = df[:, 1]

E_new = applyScatAttenuation(t, E, "CE3_Cuboid.csv", 8)

plot(t, E)
plot!(t, E_new)
savefig("Efield.pdf")