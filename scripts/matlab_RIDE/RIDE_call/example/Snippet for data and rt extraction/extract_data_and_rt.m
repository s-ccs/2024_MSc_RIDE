




%read the data 'Coil_new_seg_new_ref.vhdr' by EEGLAB
%File --> Import Data --> Using EEGLAB functions and plugins --> From Brain
%Vis. Rec. .vhdr file

%if you do not see this item, you can download it via 
%File --> Manage EEGLAB extensions --> Data import extensions

%This is an epoched data
%The time window is from -200ms to 2000ms relative to stimulus
%There are 15 different conditions, of which the names of triggers are:
%'tki','tta','pmi','tti','mip','kki','kti','mla','kta','tka','kka','lmi','pma','pmu','pfl'

%The trigger for response is 'VO_' + condition trigger

%All the information about the trigger can be found in EEG.event




%say, if i want to extract the data for the conditon 'tki':








con = 'tki';

data = [];rt = [];t=0;

%loop triggers to find 'tki'
    for k = 1:length(EEG.event)-1
        disp(k);
        if strcmpi(EEG.event(k).type,con)
            %when i found 'tki', i need to find a response trigger in the
            %folowing
            kk = k;
            latency = EEG.event(k).latency;
            while 1 %start from k until the end
                kk = kk + 1;
                if kk>length(EEG.event) break;end %if not found, break
                
                %if found, caculate the RT by the difference between kk and
                %k
                if strcmpi(EEG.event(kk).type,['VO_',con]) 
                    rt_tmp = EEG.event(kk).latency-EEG.event(k).latency;break;
                end
            end
            
            %but we need to check whether this rt (here rt_tmp) is
            %realistic or not
            %here rt is not supposed to exceed 2000ms or be less than 0
            if rt_tmp < 400 && rt_tmp >0
                t = t+1;%if satisfied add a trial
                rt(t) = rt_tmp*1000/EEG.srate;%convert to the unit of millisecond
                data(:,:,t) = EEG.data(:,:,EEG.event(k).epoch);
            end
        end
    end
    
    %now you have 'data' and 'rt' that is required for RIDE
    %for data without reaction time, simply omit the extraction of rt
    data = permute(data,[2,1,3]);%convert to the order that RIDE requires
    save(['..\',con{j},'.mat'],'data','rt');%change .. to your own directory
    
    
    %Note that there are 79 channels for this data, but not ALL of them are
    %related to brain activity. By checking the EEG.chanlocs.labels, one
    %can know the 'brain' channels are:
    %[1:57, 59:61, 79];So one should only keep the 'brain' channels before
    %applying RIDE:
    
    data = data(:,[1:57, 59:61, 79],:);
    
    
    figure;plot(linspace(EEG.xmin,EEG.xmax,size(data,1)),mean(data,3));
%the present data is a data after removal of oscular artifact but with
%articulation artifact (i.e., the response is speaking), so there is a huge
%component in the ERP

%this data only serve as example for data and rt extraction by Matlab
%scripting

