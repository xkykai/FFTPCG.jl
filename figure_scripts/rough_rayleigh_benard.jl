using Oceananigans
using Oceananigans.ImmersedBoundaries: mask_immersed_field!
using Oceananigans.Grids: xnodes, znodes
using CairoMakie
using NaNStatistics
using Printf

solver_type = get(ARGS, 1, "CG")

filename = "rough_RB_seaiceformation_noslip_periodic_halfdomain_stretched_Ra_1.0e8_Pr_1_Nr_8_Lx_1_Lz_1_Nx_1024_Nz_512_$(solver_type)"
FILE_DIR = "./Data/$(filename)"

#%%
u_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "u")
w_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "w")
T_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "T")
S_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "S")
d_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "d")
p_data = FieldTimeSeries("$(FILE_DIR)/instantaneous_fields.jld2", "p")

Nt = length(u_data.times)
times = u_data.times

xC = xnodes(u_data.grid, Center())
xF = xnodes(u_data.grid, Face())
zC = znodes(u_data.grid, Center())
zF = znodes(u_data.grid, Face())

for i in 1:Nt
    mask_immersed_field!(u_data[i], NaN)
    mask_immersed_field!(w_data[i], NaN)
    mask_immersed_field!(T_data[i], NaN)
    mask_immersed_field!(S_data[i], NaN)
    mask_immersed_field!(d_data[i], NaN)
    mask_immersed_field!(p_data[i], NaN)
end

fig = Figure(size=(1800, 1000), fontsize=20)

n = Observable(Nt)

axu = Axis(fig[1, 1]; title = "u velocity", xlabel = "x", ylabel = "z")
axw = Axis(fig[1, 3]; title = "w velocity", xlabel = "x", ylabel = "z")
axT = Axis(fig[2, 1]; title = "temperature", xlabel = "x", ylabel = "z")
axS = Axis(fig[2, 3]; title = "salinity", xlabel = "x", ylabel = "z")
axd = Axis(fig[1, 5]; title = "flow divergence", xlabel = "x", ylabel = "z")
axp = Axis(fig[2, 5]; title = "pressure", xlabel = "x", ylabel = "z")

uₙ = @lift interior(u_data[$n], :, 1, :)
wₙ = @lift interior(w_data[$n], :, 1, :)
Tₙ = @lift interior(T_data[$n], :, 1, :)
Sₙ = @lift interior(S_data[$n], :, 1, :)
dₙ = @lift interior(d_data[$n], :, 1, :)
pₙ = @lift interior(p_data[$n], :, 1, :)

ulim = (-nanmaximum(abs.(interior(u_data[Nt]))), nanmaximum(abs.(interior(u_data[Nt])))) ./ 2
wlim = (-nanmaximum(abs.(interior(w_data[Nt]))), nanmaximum(abs.(interior(w_data[Nt])))) ./ 2
Tlim = (nanminimum(interior(T_data[Nt])), nanmaximum(interior(T_data[Nt])))
Slim = (nanminimum(interior(S_data[Nt])) - 1e-4, nanmaximum(interior(S_data[Nt])) + 1e-4)
dlim = (-nanmaximum(abs.(interior(d_data[Nt]))), nanmaximum(abs.(interior(d_data[Nt])))) ./ 2
plim = (-nanmaximum(abs.(interior(p_data[Nt]))), nanmaximum(abs.(interior(p_data[Nt])))) ./ 2

hmu = heatmap!(axu, xF, zC, uₙ, colormap=:balance, colorrange=ulim)
hmw = heatmap!(axw, xC, zF, wₙ, colormap=:balance, colorrange=wlim)
hmT = heatmap!(axT, xC, zC, Tₙ, colormap=:turbo, colorrange=Tlim)
hmS = heatmap!(axS, xC, zC, Sₙ, colormap=:turbo, colorrange=Slim)
hmd = heatmap!(axd, xC, zC, dₙ, colormap=:balance, colorrange=dlim)
hmp = heatmap!(axp, xC, zC, pₙ, colormap=:balance, colorrange=plim)

Colorbar(fig[1, 2], hmu; label = "u")
Colorbar(fig[1, 4], hmw; label = "w")
Colorbar(fig[2, 2], hmT; label = "T")
Colorbar(fig[2, 4], hmS; label = "S")
Colorbar(fig[1, 6], hmd; label = "d")
Colorbar(fig[2, 6], hmp; label = "p")

time_str = @lift @sprintf("t = %.2f s", times[$n])
Label(fig[0, :], time_str, fontsize=20)

display(fig)
CairoMakie.record(fig, "./$(FILE_DIR)/$(filename).mp4", 1:Nt, framerate=10) do nn
    n[] = nn
end
#%%
Nu_data = FieldTimeSeries("$(FILE_DIR)/averaged_fields.jld2", "Nu")
Nt_averaged = length(Nu_data.times)

fig = Figure()
ax = Axis(fig[1, 1], title="Nu", xlabel="Nu", ylabel="z")
Nuₙ = interior(Nu_data[Nt_averaged], 1, 1, :)

lines!(ax, Nuₙ, znodes(Nu_data.grid, Face()))
display(fig)
save("./$(FILE_DIR)/Nu_profile.png", fig)
#%%
KE_data = FieldTimeSeries("$(FILE_DIR)/KE_fields.jld2", "KE")
Nt_averaged = length(KE_data.times)
fig = Figure()
ax = Axis(fig[1, 1], title="KE", xlabel="time", ylabel="KE")
times_averaged = KE_data.times
KE_values = interior(KE_data, 1, 1, 1, 1:Nt_averaged)
lines!(ax, times_averaged, KE_values)
display(fig)
save("./$(FILE_DIR)/KE_with_time.png", fig)
