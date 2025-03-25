set(0,'DefaultFigureWindowStyle','docked')
addpath("./RIDE_call/")

%load the sample data (/w RT) for a single subject from a face recognition task
for sec_load_data = 1:1
 load('./RIDE_call/example/samp_face.mat');
end

%configuration of parameters and run RIDE

for section = 1:1
cfg = [];%initialization
cfg.samp_interval = 2;
cfg.epoch_twd = [-100,1000];%time window for the epoched data (relative to stimulus)
cfg.comp.name = {'s','c','r'};%component names
cfg.comp.twd = {[0,500],[100,900],[-300,300]}; %time windows for extracting components, for 's' and 'c' it is raltive to stimulus, for 'r' it is relative to RT
cfg.comp.latency = {0,'unknown',rt};%latency for each RIDE component

cfg.prg = true;

cfg = RIDE_cfg(cfg);%standardize 

results = RIDE_call(data,cfg); %run RIDE
end

%Plot the time courses of RIDE results

chan_index  = find(strcmpi({chanlocs.labels},'Pz'));%select which channel to plot

%plot erp and RIDE components superimposed together
figure;RIDE_plot(results,{'erp','s','c','r'},chan_index);

%benchmark, disabled by default because it takes a while
if true == false
    cfg.prg = false;
    timed_function = @() RIDE_call(data,cfg);
    benchmark_ms = timeit(timed_function);
    disp("Benchmark result in milliseconds: " + benchmark_ms)
end