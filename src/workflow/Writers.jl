using Printf: @sprintf
using ..SimulationCore
using ..SimulationCore: virial_tensor
using ..Writers: gsd_open, gsd_close, write_gsd_frame!, read_gsd_frame!, _soa_to_posmat, _soa_to_velmat, _soa_to_frcmat, _unpermuted

"""
    Writer

Abstract workflow output writer.
"""
abstract type Writer end

"""
    TableWriter(filename; every=nothing, schedule=nothing, observables=[], mode=:replace, delimiter=",", format=:scientific, append=false, style=:auto)

Scheduled text-table writer for workflow observables.
"""
@kwdef struct TableWriter <: Writer
    filename::String
    every = nothing
    schedule = nothing
    observables::Vector{Any} = Any[]
    mode::Symbol = :replace
    delimiter::String = ","
    format::Symbol = :scientific
    append::Bool = false
    style::Symbol = :auto
end

function TableWriter(filename::AbstractString;
                     every=nothing,
                     schedule=nothing,
                     observables=Any[],
                     mode::Symbol=:replace,
                     delimiter::AbstractString=",",
                     format::Symbol=:scientific,
                     append::Bool=false,
                     style::Symbol=:auto)
    return TableWriter(filename=String(filename),
                       every=every,
                       schedule=schedule,
                       observables=Any[observables...],
                       mode=mode,
                       delimiter=String(delimiter),
                       format=format,
                       append=append,
                       style=style)
end


"""
    InMemoryAtDataLogger

Simple Logger that records the Positions and Velocities in Memory every 'every' steps or on schedule

"""
mutable struct InMemoryAtDataLogger{T<:AbstractFloat} <: Writer
    every                     # every every steps there will be output
    schedule                  # schedule for output
    expected_frames::Int      # the number of output frames to preallocate
    n_atoms::Int              # number of atoms 
    pos_data::Union{Nothing, Vector{Matrix{T}}}     # placeholder for the position Data
    vel_data::Union{Nothing, Vector{Matrix{T}}}     # placeholder for the velocity Data
    force_data::Union{Nothing, Vector{Matrix{T}}}   # placeholder for the force Data
    energy_data::Union{Nothing, Vector{Vector{T}}}  # placeholder for the energy Data
    virial_data::Union{Nothing, Vector{Matrix{T}}}  # placeholder for the virials
    current_frame::Int
    append::Bool        # dummy value, because the others have it
    mode::Symbol       # dummy value, because the others have it
end

function InMemoryAtDataLogger(; every=nothing, schedule=nothing,
                                n_atoms::Int=0, 
                                precision::Type{T} = Float32,
                                positions::Bool=false,
                                velocities::Bool=false,
                                forces::Bool=false,
                                energy::Bool=false,
                                virial::Bool=false,
                                expected_frames::Int=10) where {T<:AbstractFloat}

    if expected_frames == 0
        println("WARNING, the InMemoryAtDataLogger has been requested to be intialized with empty buffers, this may cause crashes and is at least very inefficient if the buffer is meant to be used!")
    end
        # preallocate the positions and velocities, we need to maintain a counter if we want to use this method
    pos_data    = positions  ? [Matrix{T}(undef, n_atoms, 3) for _ in 1:expected_frames] : nothing
    vel_data    = velocities ? [Matrix{T}(undef, n_atoms, 3) for _ in 1:expected_frames] : nothing
    force_data  = forces     ? [Matrix{T}(undef, n_atoms, 3) for _ in 1:expected_frames] : nothing
    energy_data = energy     ? [Vector{T}(undef, 2) for _ in 1:expected_frames]          : nothing
    virial_data = virial     ? [Matrix{T}(undef, 3, 3) for _ in 1:expected_frames]       : nothing

    # Unfortunately, we apparently need to call the positional constructor
    # my 4 argument keyword constructor is confusing the poor julia dispatch logic :(
    return InMemoryAtDataLogger{T}(every,
                       schedule,
                       Int(expected_frames),
                       Int(n_atoms),
                       pos_data, 
                       vel_data,
                       force_data,
                       energy_data,
                       virial_data,
                       1, # current frame is the index one
                       false,
                       :replace)
end

# helpers
function close_at_data_logger!(writer::InMemoryAtDataLogger)
    # cut off the undefined preallocated slots when the simulation is finished.
    n = writer.current_frame - 1
    for buf in (writer.pos_data, writer.vel_data, writer.force_data, writer.energy_data, writer.virial_data)
        buf === nothing && continue
        length(buf) > n && resize!(buf, n)
    end
    return writer
end

# Generic helper to stack any Vector of 2D matrices into a dense 3D Array: (n_frames, dim1, dim2)
function _stack_to_3d(v::Union{Nothing, Vector{Matrix{T}}}, dim::NTuple{2,Int}) where {T}
    v === nothing && return nothing
    isempty(v) && return Array{T, 3}(undef, 0, dim...)
    
    n_frames = length(v)
    res = Array{T, 3}(undef, n_frames, dim...)
    
    @inbounds for i in 1:n_frames
        res[i, :, :] = v[i]
    end
    
    return res
end
# Generic helper to stack any Vector of 1D Vectors into a dense 2D Array: (n_frames, dim1)
function _stack_to_2d(v::Union{Nothing, Vector{Vector{T}}}, dim::Int) where {T}
    v === nothing && return nothing
    isempty(v) && return Array{T, 2}(undef, 0, dim)
    
    n_frames = length(v)
    res = Array{T, 2}(undef, n_frames, dim)
    
    @inbounds for i in 1:n_frames
        res[i, :] = v[i]
    end
    return res
end

function get_and_reset_at_data_logger!(writer::InMemoryAtDataLogger)
    # for safety, return a copy of the valid parts of the buffers
    # while we are at it, convert the vector of matrices and the vector of vectors into block arrays, these are nicer generally
    last_valid_frame = writer.current_frame - 1
    
    pos_chunk = writer.pos_data === nothing ? nothing : _stack_to_3d(writer.pos_data[1:last_valid_frame], (writer.n_atoms, 3))
    vel_chunk = writer.vel_data === nothing ? nothing : _stack_to_3d(writer.vel_data[1:last_valid_frame], (writer.n_atoms, 3))
    force_chunk = writer.force_data === nothing ? nothing : _stack_to_3d(writer.force_data[1:last_valid_frame], (writer.n_atoms, 3))
    energy_chunk = writer.energy_data === nothing ? nothing : _stack_to_2d(writer.energy_data[1:last_valid_frame], 2)
    virial_chunk = writer.virial_data === nothing ? nothing : _stack_to_3d(writer.virial_data[1:last_valid_frame], (3, 3))
    
    # reset the pointer
    writer.current_frame = 1
    
    return pos_chunk, vel_chunk, force_chunk, energy_chunk, virial_chunk
end


"""
    GSDWriter(filename; every=nothing, schedule=nothing, group=Group(:all, AllSelection()), write_start=true, mode=:replace, append=false, types=:automatic, diameter=:automatic, write_unwrapped=false, sync_on_write=true, write_forces=false, write_virial=false, observables=[])

Scheduled GSD trajectory writer for workflow simulations.
"""
@kwdef struct GSDWriter <: Writer
    filename::String
    every = nothing
    schedule = nothing
    group = nothing
    write_start::Bool = true
    mode::Symbol = :replace
    append::Bool = false
    types = :automatic
    diameter = :automatic
    write_unwrapped::Bool = false
    sync_on_write::Bool = true
    write_forces::Bool = false
    write_virial::Bool = false
    observables::Vector{Any} = Any[]
end

function GSDWriter(filename::AbstractString;
                   every=nothing,
                   schedule=nothing,
                   group=Group(:all, AllSelection()),
                   write_start::Bool=true,
                   mode::Symbol=:replace,
                   append::Bool=false,
                   types=:automatic,
                   diameter=:automatic,
                   write_unwrapped::Bool=false,
                   sync_on_write::Bool=true,
                   write_forces::Bool=false,
                   write_virial::Bool=false,
                   observables=Any[])
    return GSDWriter(filename=String(filename),
                     every=every,
                     schedule=schedule,
                     group=group,
                     write_start=write_start,
                     mode=mode,
                     append=append,
                     types=types,
                     diameter=diameter,
                     write_unwrapped=write_unwrapped,
                     sync_on_write=sync_on_write,
                     write_forces=write_forces,
                     write_virial=write_virial,
                     observables=Any[observables...])
end

mutable struct PreparedWriter
    writer::Writer
    schedule
    requests::Vector{NamedTuple{(:observable, :fields),Tuple{Observable,Vector{Symbol}}}}
    handle
    header::Vector{String}
    header_written::Bool
    start_written::Bool
    needs_energy::Bool
    needs_interval_reset::Bool
    opened::Bool
end

function _normalize_writer_requests(sim, writer::Union{TableWriter})
    raw = isempty(writer.observables) ? Any[sim.observables...] : writer.observables
    isempty(raw) && throw(ArgumentError("Writers require at least one observable request."))
    requests = NamedTuple{(:observable, :fields),Tuple{Observable,Vector{Symbol}}}[]
    for entry in raw
        if entry isa Pair
            obs = entry.first
            fields = _normalize_observable_fields(obs, entry.second)
        elseif entry isa Observable
            obs = entry
            fields = _normalize_observable_fields(obs, nothing)
        else
            throw(ArgumentError("Unsupported writer observable request $(typeof(entry)). Use `observable` or `observable => [:field, ...]`."))
        end
        obs isa Observable || throw(ArgumentError("Writer requests must reference workflow Observable objects."))
        push!(requests, (observable=obs, fields=fields))
    end
    return requests
end

function _normalize_writer_requests(sim, writer::GSDWriter)
    requests = NamedTuple{(:observable, :fields),Tuple{Observable,Vector{Symbol}}}[]
    for entry in writer.observables
        if entry isa Pair
            obs = entry.first
            fields = _normalize_observable_fields(obs, entry.second)
        elseif entry isa Observable
            obs = entry
            fields = _normalize_observable_fields(obs, nothing)
        else
            throw(ArgumentError("Unsupported GSDWriter observable request $(typeof(entry))."))
        end
        push!(requests, (observable=obs, fields=fields))
    end
    return requests
end

function _normalize_writer_requests(sim, writer::InMemoryAtDataLogger)
    return NamedTuple{(:observable, :fields),Tuple{Observable,Vector{Symbol}}}[]
end

_writer_append_mode(writer::Writer) = getfield(writer, :append) || getfield(writer, :mode) == :append

function _validate_writer(writer::TableWriter)
    writer.mode in (:replace, :append) || throw(ArgumentError("TableWriter mode must be :replace or :append."))
    writer.format in (:scientific, :plain) || throw(ArgumentError("TableWriter format must be :scientific or :plain."))
    writer.style in (:auto, :csv, :log) || throw(ArgumentError("TableWriter style must be :auto, :csv, or :log."))
    return writer
end

function _validate_writer(writer::GSDWriter)
    writer.mode in (:replace, :append) || throw(ArgumentError("GSDWriter mode must be :replace or :append."))
    return writer
end

function _validate_writer(writer::InMemoryAtDataLogger)
    return writer
end


function prepare_writers!(sim)
    previous = _writer_contexts(sim)
    contexts = PreparedWriter[]
    for writer in sim.writers
        _validate_writer(writer)
        requests = _normalize_writer_requests(sim, writer)
        for req in requests
            _ensure_observable_context!(sim, req.observable)
        end
        schedule = normalize_schedule(every=getfield(writer, :every), schedule=getfield(writer, :schedule), default=Every(1))
        needs_energy = any(req -> observable_requires_energy(req.observable, req.fields), requests) ||
                       (writer isa GSDWriter && writer.write_virial)
        if writer isa InMemoryAtDataLogger #this one doesnt have requests, but bool flags instead
            needs_energy = (writer.energy_data !== nothing) || (writer.virial_data !== nothing)
        end
        needs_interval_reset = any(req -> observable_has_interval_fields(req.observable, req.fields), requests)
        ctx = PreparedWriter(writer, schedule, requests, nothing, String[], false, false, needs_energy, needs_interval_reset, false)
        old_index = findfirst(old -> old.writer === writer, previous)
        if old_index !== nothing
            old = previous[old_index]
            ctx.handle = old.handle
            ctx.header = old.header
            ctx.header_written = old.header_written
            ctx.start_written = old.start_written
            ctx.opened = old.opened
        end
        push!(contexts, ctx)
    end
    for old in previous
        any(ctx -> ctx.writer === old.writer, contexts) || _close_writer!(old)
    end
    sim.metadata[:workflow_writer_contexts] = contexts
    return sim
end

function _writer_contexts(sim)
    return get(sim.metadata, :workflow_writer_contexts, PreparedWriter[])
end

function _open_writer!(ctx::PreparedWriter)
    writer = ctx.writer
    # Replacement applies once per prepared writer session, not once per stage.
    append = ctx.opened || _writer_append_mode(writer)
    # if we have a handle, exit this
    ctx.handle === nothing || return ctx
    # else, follow the preparation, i.e. open the files (or do nothing for the InMemory Loggers)
    if writer isa TableWriter
        path = writer.filename
        mkpath(dirname(path))
        header_exists = append && isfile(path) && filesize(path) > 0
        ctx.handle = open(path, append ? "a" : "w")
        ctx.header_written = header_exists
    elseif writer isa GSDWriter
        path = writer.filename
        mkpath(dirname(path))
        if !append && isfile(path)
            rm(path; force=true)
        end
        ctx.handle = gsd_open(path; append=append)
    elseif writer isa InMemoryAtDataLogger
        ctx.handle = true # just put in a dummy val, no?
    end
    ctx.opened = true
    return ctx
end

function _close_writer!(ctx::PreparedWriter)
    if ctx.handle === nothing
        return ctx
    end
    writer = ctx.writer
    if writer isa TableWriter
        close(ctx.handle)
    elseif writer isa GSDWriter
        gsd_close(ctx.handle)
    elseif writer isa InMemoryAtDataLogger
        close_at_data_logger!(writer)
    end
    ctx.handle = nothing
    return ctx
end

function close_writers!(sim)
    for ctx in _writer_contexts(sim)
        _close_writer!(ctx)
    end
    return sim
end

function _writer_types_names(sim, writer::GSDWriter)
    if writer.types === :automatic
        return isempty(sim.system.types) ? ["A"] : String.(sim.system.types)
    elseif writer.types isa AbstractVector
        return String.(collect(writer.types))
    else
        throw(ArgumentError("GSDWriter.types must be :automatic or an AbstractVector."))
    end
end

function _writer_diameter(sim, writer::GSDWriter)
    if writer.diameter !== :automatic
        return writer.diameter
    end
    if haskey(sim.system.metadata, :diameters)
        return sim.system.metadata[:diameters]
    elseif haskey(sim.system.metadata, :diameter)
        return sim.system.metadata[:diameter]
    else
        return nothing
    end
end

function _validate_gsd_group(sim, writer::GSDWriter)
    ref = writer.group
    ref === nothing && return nothing
    if ref isa Group
        ref.selection isa AllSelection || throw(ArgumentError("GSDWriter currently supports only all-particle writes."))
    elseif ref isa Symbol
        group = sim.groups[ref]
        group.selection isa AllSelection || throw(ArgumentError("GSDWriter currently supports only all-particle writes."))
    else
        throw(ArgumentError("GSDWriter.group must be a Group or Symbol."))
    end
    return nothing
end

function _format_table_value(writer::TableWriter, value)
    if value isa AbstractFloat
        return writer.format == :scientific ? @sprintf("%.7e", value) : string(value)
    elseif value isa Real
        return string(value)
    elseif value isa Symbol
        return String(value)
    elseif value === nothing
        return ""
    else
        return string(value)
    end
end

_table_style(writer::TableWriter) =
    writer.style == :auto ? (endswith(lowercase(writer.filename), ".log") ? :log : :csv) : writer.style

_titleize_label(label::AbstractString) = replace(uppercasefirst(replace(String(label), "_" => " ")), " Msd" => " MSD", " Vacf" => " VACF")

function _workflow_group_name(obs)
    hasproperty(obs, :group) || return nothing
    group = getfield(obs, :group)
    if group isa Group
        return group.name
    elseif group isa Symbol
        return group
    end
    return nothing
end

function _group_suffix(obs)
    name = _workflow_group_name(obs)
    name === nothing && return ""
    name == :all && return ""
    return " (" * String(name) * ")"
end

function _log_field_label(obs::ThermodynamicObservable, field::Symbol)
    suffix = _group_suffix(obs)
    if field == :temperature
        return "Temperature" * suffix
    elseif field == :kinetic_energy
        return "E_kin" * suffix
    elseif field == :potential_energy
        return "E_pot" * suffix
    elseif field == :total_energy
        return "E_tot" * suffix
    elseif field == :virial
        return "virial" * suffix
    elseif field == :kinetic_energy_accumulated
        return "E_kin accum" * suffix
    elseif field == :potential_energy_accumulated
        return "E_pot accum" * suffix
    elseif field == :virial_accumulated
        return "virial accum" * suffix
    end
    return _titleize_label(field) * suffix
end

_log_field_label(::BathExchangeObservable, field::Symbol) =
    field == :heat ? "Bath Energy" :
    field == :entropy ? "Bath Entropy" :
    field == :entropy_production_rate ? "Entropy Production Rate" :
    field == :temperature_error ? "Temperature Error" :
    field == :extended_hamiltonian ? "Extended Hamiltonian" :
    field == :thermostat_kinetic ? "Thermostat Kinetic" :
    field == :thermostat_potential ? "Thermostat Potential" :
    _titleize_label(field)

function _log_field_label(obs::VirialObservable, field::Symbol)
    suffix = _group_suffix(obs)
    if field == :virial
        return "virial" * suffix
    elseif field == :virial_accumulated
        return "virial accum" * suffix
    end
    return _titleize_label(field) * suffix
end

function _log_collision_pair_label(pairkey)
    text = replace(String(pairkey), "_" => "/")
    return text * " collision rate"
end

_log_field_label(::CollisionObservable, field::Symbol) =
    field in (:counts, :rate) ? "collision rate" :
    field in (:pair_counts, :pair_rates) ? "pair collision rate" :
    _titleize_label(field)

function _log_field_label(obs::Union{MSDObservable,VACFObservable}, field::Symbol)
    suffix = _group_suffix(obs)
    return (
        field == :msd ? "MSD" * suffix :
        field == :vacf ? "VACF" * suffix :
        field == :elapsed_time ? "elapsed time" * suffix :
        field == :elapsed_steps ? "elapsed steps" * suffix :
        _titleize_label(field) * suffix
    )
end

function _log_format_value(value)
    if value isa AbstractFloat
        return @sprintf("%.5e", value)
    elseif value isa Integer
        return @sprintf("%.5e", float(value))
    elseif value isa Real
        return @sprintf("%.5e", Float64(value))
    elseif value isa Symbol
        return String(value)
    elseif value === nothing
        return ""
    else
        return string(value)
    end
end

function _center_text(text::AbstractString, width::Int)
    pad = max(0, width - length(text))
    left = pad ÷ 2
    right = pad - left
    return repeat(" ", left) * text * repeat(" ", right)
end

function _log_widths(headers::Vector{String})
    return [max(length(header), 13) + 2 for header in headers]
end

function _log_headers_and_values(sim, ctx::PreparedWriter, step::Int)
    headers = String["Time"]
    values = Any[step]
    for req in ctx.requests
        _reject_subgroup_global_virial(sim, req.observable, req.fields)
        _refresh_energies_for!(sim, req.observable, req.fields)
        data = _observable_data(sim, req.observable)
        for field in req.fields
            if req.observable isa CollisionObservable && field in (:counts, :rate)
                push!(headers, _log_field_label(req.observable, field))
                push!(values, getproperty(data, :rate))
            elseif req.observable isa CollisionObservable && field in (:pair_counts, :pair_rates)
                pair_rates = getproperty(data, :pair_rates)
                for key in _collision_pair_labels(sim)
                    push!(headers, _log_collision_pair_label(key))
                    push!(values, get(pair_rates, key, 0.0))
                end
            else
                push!(headers, _log_field_label(req.observable, field))
                push!(values, getproperty(data, field))
            end
        end
    end
    return headers, String[_log_format_value(value) for value in values]
end

function _flatten_table_value(prefix::String, value)
    if value isa NamedTuple
        headers = String[]
        values = Any[]
        for name in keys(value)
            subheaders, subvalues = _flatten_table_value(string(prefix, ".", name), getproperty(value, name))
            append!(headers, subheaders)
            append!(values, subvalues)
        end
        return headers, values
    elseif value isa AbstractDict
        headers = String[]
        values = Any[]
        for key in sort!(collect(keys(value)); by=string)
            push!(headers, string(prefix, ".", key))
            push!(values, value[key])
        end
        return headers, values
    elseif value isa AbstractVector
        headers = [string(prefix, "[", i, "]") for i in eachindex(value)]
        values = collect(value)
        return headers, values
    else
        return [prefix], [value]
    end
end

function _csv_headers_and_values(sim, ctx::PreparedWriter, step::Int)
    writer = ctx.writer::TableWriter
    headers = String["step"]
    values = Any[step]
    for req in ctx.requests
        sample = sample_observable(sim, req.observable; fields=req.fields)
        for field in req.fields
            prefix = string(observable_name(req.observable), ".", field)
            subheaders, subvalues = _flatten_table_value(prefix, getproperty(sample, field))
            append!(headers, subheaders)
            append!(values, subvalues)
        end
    end
    return headers, String[_format_table_value(writer, value) for value in values]
end

function _table_headers_and_values(sim, ctx::PreparedWriter, step::Int)
    writer = ctx.writer::TableWriter
    if _table_style(writer) == :log
        return _log_headers_and_values(sim, ctx, step)
    end
    return _csv_headers_and_values(sim, ctx, step)
end

function _write_table!(sim, ctx::PreparedWriter, step::Int)
    _open_writer!(ctx)
    writer = ctx.writer::TableWriter
    headers, values = _table_headers_and_values(sim, ctx, step)
    if !ctx.header_written
        if _table_style(writer) == :log
            widths = _log_widths(headers)
            println(ctx.handle, join((_center_text(header, widths[i]) for (i, header) in pairs(headers)), " | "))
        else
            println(ctx.handle, join(headers, writer.delimiter))
        end
        ctx.header = headers
        ctx.header_written = true
    end
    if _table_style(writer) == :log
        widths = _log_widths(headers)
        println(ctx.handle, join((_center_text(values[i], widths[i]) for i in eachindex(values)), " | "))
    else
        println(ctx.handle, join(values, writer.delimiter))
    end
    flush(ctx.handle)
    return ctx
end

function _write_gsd!(sim, ctx::PreparedWriter, step::Int)
    _open_writer!(ctx)
    writer = ctx.writer::GSDWriter
    _validate_gsd_group(sim, writer)
    write_gsd_frame!(
        ctx.handle,
        sim.state;
        diameter=_writer_diameter(sim, writer),
        types_names=_writer_types_names(sim, writer),
        step=step,
        write_forces=writer.write_forces,
        write_virial=writer.write_virial,
        write_unwrapped=writer.write_unwrapped,
        sync_on_write=writer.sync_on_write,
        logs=_writer_logs(sim),
    )
    return ctx
end

@inline function _store_frame!(buf::Vector{Matrix{T}}, idx::Int, M::AbstractMatrix) where {T}
    if idx <= length(buf) && size(buf[idx]) == size(M)
        copyto!(buf[idx], M)
    elseif idx <= length(buf)
        buf[idx] = Matrix{T}(M)
    else
        push!(buf, Matrix{T}(M))
    end
    return buf
end

@inline function _store_frame!(buf::Vector{Vector{T}}, idx::Int, M::AbstractVector) where {T}
    if idx <= length(buf) && size(buf[idx]) == size(M)
        copyto!(buf[idx], M)
    elseif idx <= length(buf)
        buf[idx] = Vector{T}(M)
    else
        push!(buf, Vector{T}(M))
    end
    return buf
end

@inline function _vir_tuple_to_mat(vir_tuple::NamedTuple)
    T = eltype(vir_tuple[:xx])
    mat = Matrix{T}([get(vir_tuple, :xx, zero(T)) get(vir_tuple, :xy, zero(T)) get(vir_tuple, :xz, zero(T));
                     get(vir_tuple, :xy, zero(T)) get(vir_tuple, :yy, zero(T)) get(vir_tuple, :yz, zero(T));
                     get(vir_tuple, :xz, zero(T)) get(vir_tuple, :yz, zero(T)) get(vir_tuple, :zz, zero(T))])
    return mat
end


function _write_at_data_inmem!(sim, ctx::PreparedWriter, step::Int)
    _open_writer!(ctx)
    writer = ctx.writer::InMemoryAtDataLogger
    
    st = sim.state
    tag = hasproperty(st, :tag) ? st.tag : nothing

    if writer.pos_data !== nothing # if they are not nothing, i.e. we have some (possibly empty buffer for the data)
        # record the positions
        posM = st.rz === nothing ? _soa_to_posmat(_unpermuted(st.rx, tag), _unpermuted(st.ry, tag)) : _soa_to_posmat(_unpermuted(st.rx, tag), _unpermuted(st.ry, tag), _unpermuted(st.rz, tag))
        _store_frame!(writer.pos_data, writer.current_frame, posM)
    end

    # and so on and so forth
    if writer.vel_data !== nothing
        velM = st.vz === nothing ? _soa_to_velmat(_unpermuted(st.vx, tag), _unpermuted(st.vy, tag)) : _soa_to_velmat(_unpermuted(st.vx, tag), _unpermuted(st.vy, tag), _unpermuted(st.vz, tag))
        _store_frame!(writer.vel_data, writer.current_frame, velM)
    end

    if writer.force_data !== nothing
        forM = st.fz === nothing ? _soa_to_frcmat(_unpermuted(st.fx, tag), _unpermuted(st.fy, tag)) : _soa_to_frcmat(_unpermuted(st.fx, tag), _unpermuted(st.fy, tag), _unpermuted(st.fz, tag))
        _store_frame!(writer.force_data, writer.current_frame, forM)
    end

    if writer.virial_data !== nothing
        # get the virial, reshape it to be a 3x3 matrix
        vir_tuple = virial_tensor(st; part=:total, accumulated=false)
        virM = _vir_tuple_to_mat(vir_tuple)
        _store_frame!(writer.virial_data, writer.current_frame, virM)
    end

    if writer.energy_data !== nothing
        ene_vec = [Base.sum(st.Ekin), Base.sum(st.Epot)]
        _store_frame!(writer.energy_data, writer.current_frame, ene_vec)
    end

    # increment the frame counter
    writer.current_frame += 1
    
    return ctx
end

function _write_writer!(sim, ctx::PreparedWriter, step::Int)
    writer = ctx.writer
    if writer isa TableWriter
        return _write_table!(sim, ctx, step)
    elseif writer isa GSDWriter
        return _write_gsd!(sim, ctx, step)
    elseif writer isa InMemoryAtDataLogger
        return _write_at_data_inmem!(sim, ctx, step)
    else
        throw(ArgumentError("Writing Attempt on currently not supported Writer?!"))
    end
end

function write_initial_frames!(sim)
    st = sim.state
    st === nothing && return sim
    for ctx in _writer_contexts(sim)
        writer = ctx.writer
        if writer isa GSDWriter && writer.write_start && !ctx.start_written
            if writer.write_forces || writer.write_virial
                spring = SimulationCore._freeze_active!(st) && st.freeze_mode == SimulationCore.FREEZE_SPRING
                SimulationCore.evaluate_current_configuration!(st, writer.write_virial; freeze_spring=spring)
            end
            _write_writer!(sim, ctx, st.step)
            ctx.start_written = true
        end
    end
    return sim
end

function write_scheduled_outputs!(sim, step::Int)
    wrote_any = false
    interval_consumed = false
    for ctx in _writer_contexts(sim)
        schedule_matches(ctx.schedule, step) || continue
        _write_writer!(sim, ctx, step)
        wrote_any = true
        interval_consumed |= ctx.needs_interval_reset
    end
    return (wrote_any=wrote_any, interval_consumed=interval_consumed)
end

function active_writer_requires_energy(sim, step::Int)
    for ctx in _writer_contexts(sim)
        schedule_matches(ctx.schedule, step) || continue
        ctx.needs_energy && return true
    end
    return false
end
