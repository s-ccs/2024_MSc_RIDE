set(0,'DefaultFigureWindowStyle','docked')
addpath("./RIDE_call/")

data_filepath

%import data
import_data_path = data_path + "simulated_data.h5";
data = h5read((import_data_path),"/dataset_data");
rt = h5read((import_data_path),"/dataset_rt");
chanlocs = load("chanlocs.mat");


%configuration of parameters and run RIDE

for section = 1:1
cfg = [];%initialization
cfg.samp_interval = 10;
cfg.epoch_twd = [-300,1600];%time window for the epoched data (relative to stimulus)
cfg.comp.name = {'s','c','r'};%component names
cfg.comp.twd = {[-200,300],[200,1000],[0,800]}; %time windows for extracting components, for 's' and 'c' it is raltive to stimulus, for 'r' it is relative to RT
cfg.comp.latency = {0,'unknown',rt*10};%latency for each RIDE component
cfg.bl =  20;
cfg.prg = 0;
% cfg.rwd = 20;
% cfg.high_cutoff = 100;

cfg = RIDE_cfg(cfg);%standardize 

results = RIDE_call(data,cfg); %run RIDE
end

%% Plot the time courses of RIDE results

%chan_index  = find(strcmpi({chanlocs.labels},'Pz'));%select which channel to plot
chan_index = 1;

%plot erp and RIDE components superimposed together
figure;RIDE_plot(results,{'erp','s','c','r'},chan_index);

%benchmark
timed_function = @() RIDE_call(data,cfg);
benchmark_ms = timeit(timed_function);
disp("Benchmark result in milliseconds: " + benchmark_ms)