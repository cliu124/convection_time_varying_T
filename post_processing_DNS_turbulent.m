clear all;
close all;

folder_path='D:\Data\dedalus';
%slurm_list={'10249087'};
slurm_list={'10424250'};
stress_list={'u_prime_u_prime',...
    'v_prime_v_prime',...
    'w_prime_w_prime',...
    'u_prime_v_prime',...
    'T_prime_T_prime',...
    'u_prime_T_prime',...
    'v_prime_T_prime',...
    'w_prime_T_prime'};
stress_label_list={'$\langle u''u''\rangle $',...
    '$\langle v''v''\rangle$',...
    '$\langle w''w''\rangle$',...
    '$\langle u''v''\rangle$',...
    '$\langle T''T''\rangle$',...
    '$\langle u''T''\rangle$',...
    '$\langle v''T''\rangle$',...
    '$\langle w''T''\rangle$'};
Re_tau=180;
% average_range=[800,931];
% average_range=[1,4000];
average_range=[90,111];
for slurm_ind=1:length(slurm_list)
    % data{slurm_ind}.y=squeeze(h5read([folder_path,'\dedalus_',slurm_list{slurm_ind},'\snapshots_channel_stress\snapshots_channel_stress_s1.h5'],['/scales/']));
    h5name=[folder_path,'\dedalus_',slurm_list{slurm_ind},'\snapshots_channel_stress\snapshots_channel_stress_s1.h5'];
    for stress_ind=1:length(stress_list)
        clear data_plot; 
        data{slurm_ind}.(stress_list{stress_ind})=squeeze(h5read(h5name,['/tasks/',stress_list{stress_ind}]));
        data{slurm_ind}.y=readDatasetByPrefix(h5name,'/scales', 'y_hash_');
        
        %h5read(h5name,'/scales/y_hash_04efb45f3573dba433867d7f01cfa48347d433e5');
    
        data_plot{1}.x=data{slurm_ind}.y;
        data_plot{1}.y=mean(data{slurm_ind}.(stress_list{stress_ind})(:,average_range),2); 
        data_plot=add_DNS_stress_literature(data_plot,stress_list{stress_ind},Re_tau);
        plot_config.label_list={1,'y',stress_label_list{stress_ind}}; 
        plot_config.Markerindex=3;
        plot_config.user_color_style_marker_list={'k-','bo'};
        plot_config.name=[folder_path,'\dedalus_',slurm_list{slurm_ind},'\',stress_list{stress_ind},'.png'];
        plot_line(data_plot,plot_config);
    end
    
end

function data = readDatasetByPrefix(filename, groupPath, prefix)
% readDatasetByPrefix  Read dataset from an HDF5 group by matching prefix
%
%   data = readDatasetByPrefix(filename, groupPath, prefix)
%
%   Example:
%       y = readDatasetByPrefix('output/data_s1.h5', '/scales', 'y_hash_');
%

    info = h5info(filename, groupPath);
    data = [];

    for i = 1:length(info.Datasets)
        dset = info.Datasets(i);
        if startsWith(dset.Name, prefix)
            dset_path = [groupPath '/' dset.Name];
            data = h5read(filename, dset_path);
            fprintf('Found dataset: %s\n', dset_path);
            return;
        end
    end

    error('No dataset starting with "%s" found in %s.', prefix, groupPath);
end

function data=add_DNS_stress_literature(data,stress_name,Re_tau)
    switch Re_tau
        case 180
            switch stress_name
                case 'u_prime_u_prime'
                    stress=[-0.9859	0.889723
                    -0.985672	1.00096
                    -0.982318	1.13996
                    -0.982047	1.27206
                    -0.978807	1.35544
                    -0.978601	1.45625
                    -0.972177	1.59521
                    -0.971942	1.70992
                    -0.968659	1.81416
                    -0.966932	1.908
                    -0.960145	2.22424
                    -0.956726	2.39453
                    -0.945776	2.49519
                    -0.940951	2.60289
                    -0.933134	2.6723
                    -0.920827	2.68604
                    -0.905528	2.66149
                    -0.887273	2.58128
                    -0.873608	2.50809
                    -0.859958	2.42795
                    -0.849356	2.35828
                    -0.837227	2.28511
                    -0.822056	2.19799
                    -0.808384	2.12827
                    -0.797761	2.06903
                    -0.784068	2.00974
                    -0.762765	1.91907
                    -0.741447	1.83534
                    -0.723178	1.76208
                    -0.686547	1.66076
                    -0.669756	1.61534
                    -0.636117	1.5523
                    -0.614721	1.50681
                    -0.582596	1.45422
                    -0.553511	1.41558
                    -0.522906	1.36996
                    -0.476981	1.31022
                    -0.440236	1.26452
                    -0.405027	1.21884
                    -0.371352	1.17318
                    -0.334622	1.12053
                    -0.299412	1.07485
                    -0.261154	1.0187
                    -0.221347	0.969474
                    -0.195325	0.93435
                    -0.166226	0.902659
                    -0.1356	0.867471
                    -0.100369	0.832218
                    -0.0620469	0.807352
                    -0.0190988	0.79285
                    0.00238946	0.792551
                    0.0576663	0.802211
                    0.105297	0.825882
                    0.152957	0.863458
                    0.183718	0.894317
                    0.211403	0.921742
                    0.237553	0.949188
                    0.265252	0.983565
                    0.29602	1.0179
                    0.336027	1.06601
                    0.388333	1.12438
                    0.426798	1.16904
                    0.472965	1.22749
                    0.522216	1.29285
                    0.577642	1.37551
                    0.611531	1.43414
                    0.639279	1.49285
                    0.670119	1.56195
                    0.696354	1.63111
                    0.719513	1.69683
                    0.758105	1.80406
                    0.778243	1.89416
                    0.796833	1.97733
                    0.812345	2.05707
                    0.830977	2.1611
                    0.843405	2.23393
                    0.860524	2.34841
                    0.88074	2.47675
                    0.900907	2.58075
                    0.922481	2.62217
                    0.943756	2.51758
                    0.951174	2.39234
                    0.957022	2.24973
                    0.962734	2.04107
                    0.968312	1.76637
                    0.975538	1.54727
                    0.981172	1.30038
                    0.986657	0.980487
                    0.990835	0.771854
                    0.996591	0.584057
                    ];
                    stress(:,2)=stress(:,2).^2;
                
                    data_len=length(data);
                    data{data_len+1}.x=stress(:,1);
                    data{data_len+1}.y=stress(:,2);
                case 'v_prime_v_prime'
                    stress=[-0.979904	0.0692455
                        -0.965934	0.145529
                        -0.950393	0.239171
                        -0.93795	0.318952
                        -0.922438	0.39869
                        -0.903842	0.485337
                        -0.882176	0.571942
                        -0.86208	0.641188
                        -0.83738	0.710369
                        -0.806575	0.762084
                        -0.774258	0.80335
                        -0.737378	0.823695
                        -0.697457	0.830092
                        -0.66062	0.82958
                        -0.614588	0.821987
                        -0.571654	0.800533
                        -0.534838	0.789592
                        -0.499579	0.768244
                        -0.453568	0.750223
                        -0.421371	0.732395
                        -0.376896	0.714395
                        -0.338559	0.696481
                        -0.300209	0.685519
                        -0.258803	0.667562
                        -0.217397	0.649605
                        -0.180574	0.64214
                        -0.140688	0.631157
                        -0.103873	0.620216
                        -0.0639659	0.619661
                        -0.0225243	0.619085
                        0.0219871	0.618466
                        0.0572964	0.621451
                        0.0956683	0.620918
                        0.135589	0.627315
                        0.177052	0.637167
                        0.218522	0.650496
                        0.256923	0.663867
                        0.299928	0.677174
                        0.336786	0.68709
                        0.372124	0.70398
                        0.41208	0.727758
                        0.453572	0.751515
                        0.488902	0.764928
                        0.528859	0.788706
                        0.571871	0.80549
                        0.607202	0.818903
                        0.647151	0.839205
                        0.684002	0.845645
                        0.725437	0.841593
                        0.765315	0.827134
                        0.800532	0.784929
                        0.832658	0.732338
                        0.861692	0.669362
                        0.883024	0.592588
                        0.902835	0.522788
                        0.922603	0.432131
                        0.937767	0.341537
                        0.949883	0.261415
                        0.961984	0.174341
                        0.97869	0.0872025
                        0.989313	0.0279586
                        ];
                        stress(:,2)=stress(:,2).^2;
                    data_len=length(data);
                    data{data_len+1}.x=stress(:,1);
                    data{data_len+1}.y=stress(:,2);
                case 'w_prime_w_prime'
                    stress=[-0.983826	0.40302
                        -0.975917	0.517626
                        -0.968022	0.62528
                        -0.947741	0.784905
                        -0.933757	0.86814
                        -0.909021	0.954702
                        -0.882821	1.00648
                        -0.841287	1.0511
                        -0.801359	1.06097
                        -0.758361	1.0708
                        -0.687828	1.03506
                        -0.652569	1.01371
                        -0.614246	0.988842
                        -0.575924	0.963976
                        -0.529949	0.928574
                        -0.497767	0.903793
                        -0.451806	0.861439
                        -0.410421	0.833053
                        -0.379773	0.808293
                        -0.344514	0.786946
                        -0.309261	0.762122
                        -0.269404	0.737234
                        -0.241827	0.712517
                        -0.197372	0.684089
                        -0.145236	0.65903
                        -0.116109	0.641244
                        -0.0624239	0.623116
                        -0.016399	0.612047
                        0.0235078	0.611492
                        0.0680405	0.621302
                        0.106434	0.631197
                        0.157127	0.651349
                        0.197077	0.671651
                        0.249319	0.698735
                        0.278525	0.719186
                        0.324642	0.753307
                        0.36462	0.787514
                        0.401528	0.821763
                        0.441506	0.85597
                        0.473788	0.879855
                        0.510697	0.914104
                        0.546063	0.944898
                        0.579887	0.972238
                        0.621393	1.00295
                        0.653675	1.02683
                        0.702862	1.06091
                        0.741277	1.08123
                        0.790421	1.09446
                        0.816514	1.09409
                        0.847169	1.07281
                        0.882378	1.02713
                        0.908372	0.978099
                        0.92512	0.911818
                        0.949429	0.789811
                        0.958503	0.723636
                        0.970562	0.615705
                        0.978072	0.535647
                        0.984041	0.452134
                        0.990024	0.375573
                        0.994465	0.295558
                        ];
                    stress(:,2)=stress(:,2).^2;
                    data_len=length(data);
                    data{data_len+1}.x=stress(:,1);
                    data{data_len+1}.y=stress(:,2);
                case 'u_prime_v_prime'
                case 'u_prime_T_prime'
                case 'v_prime_T_prime'
                case 'w_prime_T_prime'
            end
        case 550
                case 'u_prime_u_prime'
                case 'v_prime_v_prime'
                case 'w_prime_w_prime'
                case 'u_prime_v_prime'
                case 'u_prime_T_prime'
                case 'v_prime_T_prime'
                case 'w_prime_T_prime'
        otherwise 
            error('No data included');
    end


end