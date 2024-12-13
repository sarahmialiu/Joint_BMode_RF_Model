
% Main script for processing related to compounding versus flash sequence
clearvars -except mainfile vv
close all
clc


folder = "C:\Users\Samantha Gorman\Documents\Misc\Sarah\VoxelMorph\Jad_RFData";
subfolders = dir(folder);

for i = 1:length(subfolders)
    scans = dir(folder + "\" + subfolders(i).name);
    for j = 1:length(scans)
        files = dir(folder + "\" + subfolders(i).name + "\" + scans(j).name);
        disp(subfolders(i).name)
        disp(scans(j).name)
        for k = 1:length(files)
            numTX = 3;
            if contains(files(k).name, ["compounding_fullap_", "rf_data_"])
                load(folder+"\"+subfolders(i).name+"\"+scans(j).name+"\"+files(k).name)
                rf_data_pad = rf_data;
                if size(rf_data_pad,3) == 30
                    numTX = 1;
                    disp("found uncompounded data!")
                end
                rfScanConv = generateRFScanConv(rf_data_pad, tstart, numTX);
                save(folder + "\" + subfolders(i).name + "\" + scans(j).name + '\ConvRF','rfScanConv')
            elseif contains(files(k).name, "rf_data_aberrated")
                load(folder+"\"+subfolders(i).name+"\"+scans(j).name+"\"+files(k).name)
                rf_data_pad = aberrated_rf;
                if size(rf_data_pad,3) == 30
                    numTX = 1;
                    disp("found uncompounded data!")
                end
                rfScanConv = generateRFScanConv(rf_data_pad, tstart, numTX);
                save(folder + "\" + subfolders(i).name + "\" + scans(j).name + '\ConvRF','rfScanConv')
            end
        end
    end
end

% load('E:\VoxelMorph\Jad_RFData\1tx\6mmgap_5mmdeep_1\rf_data_aberrated1_TX3_10-Feb-2020.mat');
% 
% %% Add noise to RF
% rf_data_pad=aberrated_rf;

