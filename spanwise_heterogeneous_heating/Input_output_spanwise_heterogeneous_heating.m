% Code by Jino George . Group of Dr. Chang Liu University of Connecticut

clc; close all; clear all;

Re = 350;
Ri = 1;
Pr=0.71;

Ny = 32; Nz = 24;
N = Ny*Nz;
c_number = 1; % change from 96 to 12
%grad_P = -2/Re*ones(N,1); % pressure gradient, vector not matrix

kxn = 1; %sample points of kx. 
Lz=2*pi; %spanwise domain size. In default, we have Lz=2pi

% omega = 1; % frequency
kx_list = logspace(-4,0.48,kxn);
c_list = linspace(0,1,c_number); % from 0 to 1

% kx_list=0;
% c_list=0;

%get Fourier differentiation matrix
[z,Dz] = fourdif(Nz,1); % first derivative z
[~,Dzz] = fourdif(Nz,2); % second derivative z

%scale the derivative matrix and the domain length
Dz = Dz*(2*pi/Lz);
Dzz = Dzz*(2*pi/Lz)^2;
z = z * (Lz / (2*pi));
Iz = speye(size(Dzz)); %sparse identity matrix

%get Chebyshev differentiation matrix, here in default is within [-1,1] and
%do not need to scale it. 
[y,DM] = chebdif(Ny,2);
D1 = DM(:,:,1);
D2 = DM(:,:,2);
Iy = speye(size(D2)); %sparse identity matrix

%Change this to your Dedalus data path for base state. 
mean_data_path='\\wsl.localhost\Ubuntu\home\changliu\convection_time_varying_T\spanwise_heterogeneous_heating\snapshots_channel\snapshots_channel_s1.h5';

%read x, y, z, and t coordinates for reading data. 
x_dedalus=readDatasetByPrefix(mean_data_path,'/scales', 'x_hash_');
y_dedalus=readDatasetByPrefix(mean_data_path,'/scales', 'y_hash_');
z_dedalus=readDatasetByPrefix(mean_data_path,'/scales', 'z_hash_');
t=h5read(mean_data_path,'/scales/sim_time');

%read the u(x,y,z) at the last time step from Dedalus data
U_all_mean=h5read(mean_data_path,'/tasks/u',[1,1,1,1,length(t)],[length(z_dedalus),length(y_dedalus),length(x_dedalus),3,1]);

%Get each component of velocity and perform average in x. 
mean_dedalus.u=mean(U_all_mean(:,:,:,1,end),3);
mean_dedalus.v=mean(U_all_mean(:,:,:,2,end),3);
mean_dedalus.w=mean(U_all_mean(:,:,:,3,end),3);

field_list={'dudy','dudz','dvdy','dvdz','dwdy','dwdz','dTdy','dTdz'};

for field_ind=1:length(field_list)
    field_name=field_list{field_ind};

    %read the 3D spatial data at the last time step from Dedalus data
    mean_dedalus.(field_name)=h5read(mean_data_path,['/tasks/',field_name],[1,1,1,length(t)],[length(z_dedalus),length(y_dedalus),length(x_dedalus),1]);

    %perform average over streamwise and take the moment in time.
    mean_dedalus.(field_name)=mean(mean_dedalus.(field_name)(:,:,:,end),3);
end

%interpolate data to new mesh obtained from fourdif and chebdif. They are
%in a different mesh from Dedalus!
[Y_old, Z_old] = meshgrid(y_dedalus,z_dedalus);
[Y_new, Z_new] = meshgrid(y,z);  % size (Nx,Ny)

mean_IO=struct;
mean_IO_vec=struct;
fn = fieldnames(mean_dedalus);
for k = 1:numel(fn)
    field=fn{k};

    %interpolation
    mean_IO.(field)=interp2(Y_old,Z_old,mean_dedalus.(field),Y_new,Z_new,'linear');

    %transpose to make different rows are different y, and vectorize it so
    %it can be directly used in input-output analysis. 
    mean_IO_vec.(field)=reshape(transpose(mean_IO.(field)),Ny*Nz,1);

    %make the diagonal matrix a sparse objective. 
    mean_IO_vec_diag.(field)=spdiags(mean_IO_vec.(field),0,N,N);

end

[x,w] = clencurt(Ny-1);
w = kron(Iz,spdiags(w',0,Ny,Ny));
weight=blkdiag(w.^0.5,w.^0.5,w.^0.5,w.^0.5);
inv_weight=inv(weight);

I = speye(Ny*Nz); %sparse identity matrix
Z=sparse(N,N); %sparse zero matrix

B=[I,Z,Z,Z;
   Z,I,Z,Z;
   Z,Z,I,Z;
   Z,Z,Z,Z;
   Z,Z,Z,I];

C=[I,Z,Z,Z,Z;
   Z,I,Z,Z,Z;
   Z,Z,I,Z,Z;
   Z,Z,Z,Z,I];

E=blkdiag(I,I,I,Z,I);

% 
% delete(gcp("nocreate"));
% parpool(8); % EQUAL TO --NTASKS IN CLUSTER

%Differentiation matrix for 2D problem. 
DM_2D.Dy=kron(Iz,D1);
DM_2D.Dyy=kron(Iz,D2);
DM_2D.Dz=kron(Dz,Iy);
DM_2D.Dzz=kron(Dzz,Iy);

for kx_ind=1:length(kx_list) % 48

    kx = kx_list(kx_ind);

    %laplacian
    laplacian = -kx^2*kron(Iz,Iy) + DM_2D.Dyy + DM_2D.Dzz;

    %advection term that is common term in diagonal of A matrix.
    L_A = -mean_IO_vec_diag.u*1i*kx-mean_IO_vec_diag.v*DM_2D.Dy-mean_IO_vec_diag.w*DM_2D.Dz;

    %u momentum equation
    A11 = L_A + 1/Re.*laplacian;
    A12 = -mean_IO_vec_diag.dudy;
    A13 = -mean_IO_vec_diag.dudz;
    A14 = -1i*kx*I;
    A15 = Z; %sparse zero matrix

    %v momentum equation
    A21 = Z;
    A22 = L_A + 1/Re*laplacian-mean_IO_vec_diag.dvdy;
    A23 = -mean_IO_vec_diag.dvdz;
    A24 = -DM_2D.Dy;
    A25 = Ri*I;

    %w momentum equation
    A31 = Z;
    A32 = -mean_IO_vec_diag.dwdy;
    A33 = L_A + 1/Re*laplacian-mean_IO_vec_diag.dwdz;
    A34 = -DM_2D.Dz;
    A35 = Z;

    %continuity equation
    A41 = 1i*kx*I;
    A42 = DM_2D.Dy;
    A43 = DM_2D.Dz;
    A44 = Z;
    A45 = Z;

    %temperature equation
    A51 = Z;
    A52 = -mean_IO_vec_diag.dTdy+0.5*I; %change the sign before 0.5*I to make it unstable stratified. 
    A53 = -mean_IO_vec_diag.dTdz;
    A54 = Z;
    A55 = L_A + 1/(Re*Pr)*laplacian;

    A = [A11 A12 A13 A14 A15; 
        A21 A22 A23 A24 A25; 
        A31 A32 A33 A34 A35; 
        A41 A42 A43 A44 A45; 
        A51 A52 A53 A54 A55];

    %vectorized implementation of the boundary condition

    
    % Boundary row indices (3) vectorized
    topRows    = (0:Nz-1)*Ny + 1;
    bottomRows = (1:Nz)*Ny;
    bd = [topRows, bottomRows];

    % (3) Vectorized boundary enforcement on A, and zero same rows in B & E
    %bd_all = [bd, bd+N, bd+2*N];  % u,v,w rows in 4N system
    % zero rows then set diagonal entry to 1 on those rows & their variable blocks
    A(bd, :)        = 0;  A(sub2ind([5*N,5*N], bd, bd)) = 1;
    A(bd+N, :)      = 0;  A(sub2ind([5*N,5*N], bd+N, bd+N)) = 1;
    A(bd+2*N, :)    = 0;  A(sub2ind([5*N,5*N], bd+2*N, bd+2*N)) = 1;
    A(bd+4*N, :)    = 0;  A(sub2ind([5*N,5*N], bd+4*N, bd+4*N)) = 1;
  
    B(bd, :)        = 0;  B(bd+N, :) = 0;  B(bd+2*N, :) = 0; B(bd+4*N, :) = 0;
    E(bd, :)        = 0;  E(bd+N, :) = 0;  E(bd+2*N, :) = 0; E(bd+4*N, :) = 0;


    % Boundary Conditions
    % 
    % for z_ind = 0:Nz-1
    %     % u Boundary Condition
    %     % u(y=1,z)=0
    %     A(1+z_ind*Ny,:) = 0;
    %     A(1+z_ind*Ny, 1+z_ind*Ny:(z_ind+1)*Ny) = [1,zeros(1,Ny-1)];  %+ Ls_u*D1(1,:); %This should be +Ls, because on the top wall y=1, B.C. should be u_s=-Ls*du/dy
    % 
    %     % u(y=-1,z)=0
    %     A(Ny+z_ind*Ny,:) = 0;
    %     A(Ny+z_ind*Ny, 1+z_ind*Ny:(z_ind+1)*Ny) = [zeros(1,Ny-1),1]; %- Ls_u*D1(N,:); %at bottom wall y=-1, B.C. is u_s=Ls*du/dy
    % 
    %     % V Boundary Condition
    %     % v(y=1,z)=0
    %     A(N+1+z_ind*Ny, :) = 0;
    %     A(N+1+z_ind*Ny, N+1+z_ind*Ny:N+(z_ind+1)*Ny) = [1,zeros(1,Ny-1)];  %+ Ls_u*D1(1,:); %This should be +Ls, because on the top wall y=1, B.C. should be u_s=-Ls*du/dy
    % 
    %     % v(y=-1,z)=0
    %     A(N+Ny+z_ind*Ny, :) = 0;
    %     A(N+Ny+z_ind*Ny, N+1+z_ind*Ny:N+(z_ind+1)*Ny) = [zeros(1,Ny-1),1]; %- Ls_u*D1(N,:); %at bottom wall y=-1, B.C. is u_s=Ls*du/dy
    % 
    %     % W Boundary Condition
    %     % w(y=1,z)=0
    %     A(2*N+1+z_ind*Ny, :) = 0;
    %     A(2*N+1+z_ind*Ny, 2*N+1+z_ind*Ny:2*N+(z_ind+1)*Ny) = [1,zeros(1,Ny-1)];  %+ Ls_u*D1(1,:); %This should be +Ls, because on the top wall y=1, B.C. should be u_s=-Ls*du/dy
    % 
    %     % w(y=-1,z)=0
    %     A(2*N+Ny+z_ind*Ny, :) = 0;
    %     A(2*N+Ny+z_ind*Ny, 2*N+1+z_ind*Ny:2*N+(z_ind+1)*Ny ) = [zeros(1,Ny-1),1]; %- Ls_u*D1(N,:); %at bottom wall y=-1, B.C. is u_s=Ls*du/dy
    % 
    %     %T boundary condition
    %     % theta(y=1,z)=0
    %     A(4*N+1+z_ind*Ny, :) = 0;
    %     A(4*N+1+z_ind*Ny, 4*N+1+z_ind*Ny:4*N+(z_ind+1)*Ny) = [1,zeros(1,Ny-1)];  %+ Ls_u*D1(1,:); %This should be +Ls, because on the top wall y=1, B.C. should be u_s=-Ls*du/dy
    % 
    %     % theta(y=-1,z)=0. Here, non-homogeneous term already been
    %     % incorporated into the base state, such as horizontal convection
    %     % roll. 
    %     A(4*N+Ny+z_ind*Ny, :) = 0;
    %     A(4*N+Ny+ z_ind*Ny, 4*N+1+z_ind*Ny:4*N+(z_ind+1)*Ny) = [zeros(1,Ny-1),1]; %- Ls_u*D1(N,:); %at bottom wall y=-1, B.C. is u_s=Ls*du/dy
    % 
    %     % B Matrix
    %     B(1 + z_ind*Ny,:) = 0;
    %     B(Ny+ z_ind*Ny,:) = 0;
    %     B(N+1+z_ind*Ny,:) = 0;
    %     B(N+Ny+ z_ind*Ny,:) = 0;
    %     B(2*N+1+z_ind*Ny,:) = 0;
    %     B(2*N+Ny+ z_ind*Ny,:) = 0;
    %     B(4*N+1+z_ind*Ny,:) = 0;
    %     B(4*N+Ny+z_ind*Ny,:) = 0;
    % 
    %     % E Matrix
    %     E(1 + z_ind*Ny,:) = 0;
    %     E(Ny+ z_ind*Ny,:) = 0;
    %     E(N+1+z_ind*Ny,:) = 0;
    %     E(N+Ny+ z_ind*Ny,:) = 0;
    %     E(2*N+1+z_ind*Ny,:) = 0;
    %     E(2*N+Ny+ z_ind*Ny,:) = 0;
    %     E(4*N+1+z_ind*Ny,:) = 0;
    %     E(4*N+Ny+ z_ind*Ny,:) = 0;
    % 
    % end


    C_tilde = weight*C ;
    B_tilde = B*inv_weight;

    for c_index=1:length(c_list)

        omega = -c_list(c_index)*kx;

        H = C_tilde*inv(E*1i*omega-A)*B_tilde;

        [U_svd,sigma,V_svd] = svds(H,1);

        result_sigma(c_index,kx_ind) = sigma(1,1);

        result_U_svd{c_index,kx_ind} = inv_weight*U_svd(:,1);
        result_V_svd{c_index,kx_ind} = inv_weight*V_svd(:,1);

    end

    kx_ind

end


% [U_svd,Sigma,V_svd] --> U is the response mode and V is the forcing mode
U_hat = real(reshape(result_U_svd{1,1}(1:N),Ny,Nz)); % Response mode
V_hat = real(reshape(result_U_svd{1,1}(N+1:2*N),Ny,Nz));
W_hat = real(reshape(result_U_svd{1,1}(2*N+1:3*N),Ny,Nz));
T_hat = real(reshape(result_U_svd{1,1}(3*N+1:4*N),Ny,Nz));

fx_hat = real(reshape(result_V_svd{1,1}(1:N),Ny,Nz)); % Response mode
fy_hat = real(reshape(result_V_svd{1,1}(N+1:2*N),Ny,Nz));
fz_hat = real(reshape(result_V_svd{1,1}(2*N+1:3*N),Ny,Nz));
fT_hat = real(reshape(result_V_svd{1,1}(3*N+1:4*N),Ny,Nz));

save('IO_spanwise_heterogeneity.mat');


% figure(1)
% pcolor(real(result_sigma))
% title('Singular Value Decomposition')
% hold on
%
% figure(2)
% plot(max(result_sigma))
% % colormap(jet)
% % shading interp
% title('Maximum Singular Value Decomposition') % Find the peak of SVD and find the
% % corresponding U and V values of SVD.
% hold on

figure(3)
pcolor(z,y,U_hat);
colormap(jet)
shading interp
title('U response mode')
hold on

figure(4)
pcolor(z,y,V_hat);
colormap(jet)
shading interp
title('V response mode')
hold on

figure(5)
pcolor(z,y,W_hat)
colormap(jet)
shading interp
title('W response mode')
hold on

figure(6)
pcolor(z,y,T_hat)
colormap(jet)
shading interp
title('T response mode')
hold on


figure(7)
pcolor(z,y,fx_hat)
colormap(jet)
shading interp
title('U forcing mode')
hold on

figure(8)
pcolor(z,y,fy_hat)
colormap(jet)
shading interp
title('V forcing mode')
hold on

figure(9)
pcolor(z,y,fz_hat)
colormap(jet)
shading interp
title('W forcing mode')
hold on


figure(10)
pcolor(z,y,fT_hat)
colormap(jet)
shading interp
title('T forcing mode')
hold on

% Forcing modes and response modes by performing SVD . The singular value
% is the amplification matrix. we use the volume penallty method to
% understand flow over wavy structure. This is in spanwise direction

% We are generating small scare structures or wave number is higher than
% the length o fthe structure.


% figure(1)
% pcolor(kz_list,kx_list,log10(Hinf));
% set(gca,'xscale','log');set(gca,'yscale','log');
% colormap(jet);
% shading interp
% caxis([1.5,4.0]);
% colorbar;
% xlabel('kz'); ylabel('kz');
% hold on
%
%
% figure(2)
% pcolor(kz_list,kx_list,log10(mu_max));
% set(gca,'xscale','log');set(gca,'yscale','log');
% colormap(jet);
% shading interp
% colorbar;
% xlabel('kz'); ylabel('kz');
% caxis([1.0,3.0]);
%
%
% figure(3)
% pcolor(kz_list,kx_list,log10(Hinf_grad));
% set(gca,'xscale','log');set(gca,'yscale','log');
% colormap(jet);
% shading interp
% colorbar;
% xlabel('kz'); ylabel('kz');



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