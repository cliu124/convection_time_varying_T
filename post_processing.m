clear all;
close all;
clc;


omega_list=[0.1,0.2,0.5,1,2,5,10,20];

slurm_list={'16473707',
    '16473708',
    '16473709',
    '16473711',
    '16473713',
    '16473717',
    '16473719',
    '16473721'};

for slurm_ind=1:length(slurm_list)
    dTdy{slurm_ind}=squeeze(h5read(['D:\Data\dedalus\dedalus_',slurm_list{slurm_ind},'\snapshots_channel_thermal\snapshots_channel_thermal_s1.h5'],'/tasks/dTdy'));
    t=h5read(['D:\Data\dedalus\dedalus_',slurm_list{slurm_ind},'\snapshots_channel_thermal\snapshots_channel_thermal_s1.h5'],'/scales/sim_time');
    max_dTdy(slurm_ind)=max(max(abs(dTdy{slurm_ind}(1,100:end))));
    mean_dTdy(slurm_ind)=mean(dTdy{slurm_ind}(1,200:end));
    omega=omega_list(slurm_ind);
    T=1+0.5*sin(omega*t);
    h{slurm_ind}=squeeze(dTdy{slurm_ind}(1,:))./T';
    h_mean(slurm_ind)=mean(h{slurm_ind});
end

data{1}.x=omega_list;
data{1}.y=max_dTdy;
plot_config.loglog=[1,1];
plot_config.label_list={1,'$\omega$','$max(dT/dz)$'};
plot_config.print_size=[1,1000,600];
plot_config.Markerindex=2;
plot_config.xlim_list=[1,0.1,20];
plot_config.xtick_list=[1,0.1,1,10];
plot_config.name='max_dTdz.png';
plot_line(data,plot_config);

data{1}.x=omega_list;
data{1}.y=abs(mean_dTdy);
plot_config.loglog=[1,0];
plot_config.Markerindex=2;
plot_config.label_list={1,'$\omega$','$\overline{dT/dz}$'};
plot_config.print_size=[1,1000,600];
plot_config.xlim_list=[1,0.1,20];
plot_config.xtick_list=[1,0.1,1,10];
plot_config.name='mean_dTdz.png';
plot_line(data,plot_config);