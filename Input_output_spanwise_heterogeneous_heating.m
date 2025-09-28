% Code by Jino George . Group of Dr. Chang Liu University of Connecticut

clc; close all; clear all;

%N = 30;
Re = 690;
Ri = 1;
Pr=0.71;

Ny = 60; Nz = 45;
N = Ny*Nz;
c_number = 1; % change from 96 to 12
grad_P = -2/Re*ones(N,1); % pressure gradient, vector not matrix

kxn = 1;

Lz=2*pi;

omega = 1; % frequency
kx_list = logspace(-4,0.48,kxn);
%kz_list = logspace(-2,1.2,kzn);

c_list = linspace(0,1,c_number); % from 0 to 1
% kx_list = 1;
% kz_list = 1;

% wavenumber. write for loop at end of iteration
%kx = 1; kz = 1;
% i = 1;j = 1;

%get Fourier differentiation matrix
[~,Dzz] = fourdif(Nz,2); % second derivative z
[z,Dz] = fourdif(Nz,1); % first derivative z

%scale the derivative matrix and the domain length
Dz = Dz*(2*pi/Lz);
Dzz = Dzz*(2*pi/Lz)^2;
z = z * (Lz / (2*pi));
Iz = eye(size(Dzz));

%U = zeros(N);
[y,DM] = chebdif(Ny,2);
D2 = DM(:,:,2);
D1 = DM(:,:,1);
Iy = eye(size(D2));

mean_data_path='\\wsl.localhost\Ubuntu\home\changliu\convection_time_varying_T\snapshots_channel\snapshots_channel_s1.h5';
U_all_mean=h5read(mean_data_path,'/tasks/u');

mean_dedalus.u=mean(U_all_mean(:,:,:,1,end),3);
mean_dedalus.v=mean(U_all_mean(:,:,:,2,end),3);
mean_dedalus.w=mean(U_all_mean(:,:,:,3,end),3);

field_list={'dudy','dudz','dvdy','dvdz','dwdy','dwdz','dTdy','dTdz'};

for field_ind=1:length(field_list)
    field_name=field_list{field_ind};
    mean_dedalus.(field_name)=h5read(mean_data_path,['/tasks/',field_name]);

    %perform average over streamwise and take the moment in time.
    mean_dedalus.(field_name)=mean(mean_dedalus.(field_name)(:,:,:,end),3);
end

y_dedalus=readDatasetByPrefix(mean_data_path,'/scales', 'y_hash_');
z_dedalus=readDatasetByPrefix(mean_data_path,'/scales', 'z_hash_');

[Y_old, Z_old] = meshgrid(y_dedalus,z_dedalus);
[Y_new, Z_new] = meshgrid(y,z);  % size (Nx,Ny)

mean_IO=struct;
mean_IO_vec=struct;
fn = fieldnames(mean_dedalus);
for k = 1:numel(fn)
    field=fn{k};

    %interpolation
    mean_IO.(field)=interp2(Y_old,Z_old,mean_dedalus.(field),Y_new,Z_new,'linear');

    %transpose to make different rows are different y, and vectorize to
    mean_IO_vec.(field)=reshape(transpose(mean_IO.(field)),Ny*Nz,1);
end

[x,w] = clencurt(Ny-1);
w = kron(Iz,diag(w));

A = zeros(5*N); B = zeros(5*N,4*N); C = zeros(4*N,5*N); E = zeros(5*N);

A11 = zeros(N); A12 = zeros(N); A13 = zeros(N); A14 = zeros(N); A15 = zeros(N);
A21 = zeros(N); A22 = zeros(N); A23 = zeros(N); A24 = zeros(N); A25 = zeros(N);
A31 = zeros(N); A32 = zeros(N); A33 = zeros(N); A34 = zeros(N); A35 = zeros(N);
A41 = zeros(N); A42 = zeros(N); A34 = zeros(N); A44 = zeros(N); A45 = zeros(N);
A51 = zeros(N); A52 = zeros(N); A53 = zeros(N); A54 = zeros(N); A55 = zeros(N);

% 
% delete(gcp("nocreate"));
% parpool(8); % EQUAL TO --NTASKS IN CLUSTER

I = eye(Ny*Nz);

DM_2D.Dy=kron(Iz,D1);
DM_2D.Dyy=kron(Iz,D2);
DM_2D.Dz=kron(Dz,Iy);
DM_2D.Dzz=kron(Dzz,Iy);

for i=1:kxn % 48

    kx = kx_list(i);

    %laplacian
    laplacian = -kx^2*kron(Iz,Iy) + DM_2D.Dyy + DM_2D.Dzz;

    %advection term that is common term in diagonal of A matrix.
    L_A = -diag(mean_IO_vec.u)*1i*kx-diag(mean_IO_vec.v)*DM_2D.Dy-diag(mean_IO_vec.w)*DM_2D.Dz;

    % Interior Matrix

    %u momentum equation
    A11 = L_A + 1/Re.*laplacian;
    A12 = -diag(mean_IO_vec.dudy);
    A13 = -diag(mean_IO_vec.dudz);
    A14 = -1i*kx*I;

    %v momentum equation
    A22 = L_A + 1/Re*laplacian-diag(mean_IO_vec.dvdy);
    A23 = -diag(mean_IO_vec.dvdz);
    A24 = -DM_2D.Dy;
    A25 = Ri*I;

    %w momentum equation
    A32 = -diag(mean_IO_vec.dwdy);
    A33 = L_A + 1/Re*laplacian-diag(mean_IO_vec.dwdz);
    A34 = -DM_2D.Dz;

    %continuity equation
    A41 = 1i*kx*I;
    A42 = DM_2D.Dy;
    A43 = DM_2D.Dz;

    %temperature equation
    A52 = -diag(mean_IO_vec.dTdy)-0.5*I;
    A53 = -diag(mean_IO_vec.dTdz);
    A55 = L_A + 1/(Re*Pr)*laplacian;

    A = [A11 A12 A13 A14 A15; A21 A22 A23 A24 A25; A31 A32 A33 A34 A35; A41 A42 A43 A44 A45; A51 A52 A53 A54 A55];


    B(1:N,1:N) = I;
    B(N+1:2*N,N+1:2*N) = I;
    B(2*N+1:3*N,2*N+1:3*N) = I;
    B(4*N+1:5*N,3*N+1:4*N) = I;

    E(1:N,1:N) = I;
    E(N+1:2*N,N+1:2*N) = I;
    E(2*N+1:3*N,2*N+1:3*N) = I;
    E(4*N+1:5*N,4*N+1:5*N) = I;

    C(1:N,1:N) = I;
    C(N+1:2*N,N+1:2*N) = I;
    C(2*N+1:3*N,2*N+1:3*N) = I;
    C(3*N+1:4*N,4*N+1:5*N) = I;

    % Boundary Conditions

    for z_ind = 0:Nz-1
        % U Boundary Condition
        A(1 + z_ind*Ny,:) = 0;
        %A(1,1) = 1;

        A(1+ z_ind*Ny,1+z_ind*Ny:(z_ind+1)*Ny) = [1,zeros(1,Ny-1)];  %+ Ls_u*D1(1,:); %This should be +Ls, because on the top wall y=1, B.C. should be u_s=-Ls*du/dy

        A(Ny+ z_ind*Ny,:) = 0;
        %A(N,N) = 1;
        %A(1,2:4*N) = 0; % U boundary condition
        A(Ny+ z_ind*Ny,1 + z_ind*Ny:(z_ind+1)*Ny ) = [zeros(1,Ny-1),1]; %- Ls_u*D1(N,:); %at bottom wall y=-1, B.C. is u_s=Ls*du/dy

        % V Boundary Condition
        A(N+1+z_ind*Ny,:) = 0;
        %A(1,1) = 1;

        A(N+1+z_ind*Ny,N+1+z_ind*Ny:N+(z_ind+1)*Ny) = [1,zeros(1,Ny-1)];  %+ Ls_u*D1(1,:); %This should be +Ls, because on the top wall y=1, B.C. should be u_s=-Ls*du/dy

        A(N+Ny+ z_ind*Ny,:) = 0;
        %A(N,N) = 1;
        %A(1,2:4*N) = 0; % U boundary condition
        A(N+Ny+ z_ind*Ny,N+1+z_ind*Ny:N+(z_ind+1)*Ny ) = [zeros(1,Ny-1),1]; %- Ls_u*D1(N,:); %at bottom wall y=-1, B.C. is u_s=Ls*du/dy

        % W Boundary Condition
        A(2*N+1+z_ind*Ny,:) = 0;
        %A(1,1) = 1;

        A(2*N+1+z_ind*Ny,2*N+1+z_ind*Ny:2*N+(z_ind+1)*Ny) = [1,zeros(1,Ny-1)];  %+ Ls_u*D1(1,:); %This should be +Ls, because on the top wall y=1, B.C. should be u_s=-Ls*du/dy

        A(2*N+Ny+ z_ind*Ny,:) = 0;

        A(2*N+Ny+ z_ind*Ny,2*N+1+z_ind*Ny:2*N+(z_ind+1)*Ny ) = [zeros(1,Ny-1),1]; %- Ls_u*D1(N,:); %at bottom wall y=-1, B.C. is u_s=Ls*du/dy

        %T boundary condition
        % theta'(y=1)=0
        A(4*N+1+z_ind*Ny,:) = 0;
        A(4*N+1+z_ind*Ny,4*N+1+z_ind*Ny:4*N+(z_ind+1)*Ny) = [1,zeros(1,Ny-1)];  %+ Ls_u*D1(1,:); %This should be +Ls, because on the top wall y=1, B.C. should be u_s=-Ls*du/dy

        % theta'(y=-1)=0 for now. This is non-homogeneous and needs to be
        % updated!!!!!
        A(4*N+Ny+z_ind*Ny,:) = 0;
        A(4*N+Ny+ z_ind*Ny,4*N+1+z_ind*Ny:4*N+(z_ind+1)*Ny) = [zeros(1,Ny-1),1]; %- Ls_u*D1(N,:); %at bottom wall y=-1, B.C. is u_s=Ls*du/dy

        % B Matrix
        B(1 + z_ind*Ny,:) = 0;
        B(Ny+ z_ind*Ny,:) = 0;
        B(N+1+z_ind*Ny,:) = 0;
        B(N+Ny+ z_ind*Ny,:) = 0;
        B(2*N+1+z_ind*Ny,:) = 0;
        B(2*N+Ny+ z_ind*Ny,:) = 0;
        B(4*N+1+z_ind*Ny,:) = 0;
        B(4*N+Ny+z_ind*Ny,:) = 0;


        % E Matrix
        E(1 + z_ind*Ny,:) = 0;
        E(Ny+ z_ind*Ny,:) = 0;
        E(N+1+z_ind*Ny,:) = 0;
        E(N+Ny+ z_ind*Ny,:) = 0;
        E(2*N+1+z_ind*Ny,:) = 0;
        E(2*N+Ny+ z_ind*Ny,:) = 0;
        E(4*N+1+z_ind*Ny,:) = 0;
        E(4*N+Ny+ z_ind*Ny,:) = 0;

    end


    C_tilde = blkdiag(w.^0.5,w.^0.5,w.^0.5,w.^0.5)*C ;
    B_tilde = B* inv(blkdiag(w.^(0.5),w.^(0.5),w.^(0.5),w.^(0.5)));


    for c_index=1:c_number

        omega = -c_list(c_index)*kx;

        H = C*inv(E*1i*omega-A)*B;

        [U_svd,sigma,V_svd] = svd(H);

        result_sigma(c_index,i) = sigma(1,1);

        result_U_svd{c_index,i} = U_svd(:,1);
        result_V_svd{c_index,i} = V_svd(:,1);

    end

    i

end


% [U_svd,Sigma,V_svd] --> U is the response mode and V is the forcing mode
U_hat = real(reshape(result_U_svd{1,1}(1:N),Nz,Ny)); % Response mode
V_hat = real(reshape(result_U_svd{1,1}(N+1:2*N),Nz,Ny));
W_hat = real(reshape(result_U_svd{1,1}(2*N+1:3*N),Nz,Ny));

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
%
% figure(3)
% pcolor(z,y,real(reshape(result_U_svd{1,1}(1:N),Ny,Nz)))
% colormap(jet)
% shading interp
% title('U response mode')
% hold on
%
% figure(4)
% pcolor(z,y,real(reshape(result_U_svd{1,1}(N+1:2*N),Ny,Nz)))
% colormap(jet)
% shading interp
% title('V response mode')
% hold on
%
% figure(5)
% pcolor(z,y,real(reshape(result_U_svd{1,1}(2*N+1:3*N),Ny,Nz)))
% colormap(jet)
% shading interp
% title('W response mode')
% hold on
%
%
% figure(6)
% pcolor(z,y,real(reshape(result_V_svd{1,1}(1:N),Ny,Nz)))
% colormap(jet)
% shading interp
% title('U forcing mode')
% hold on
%
% figure(7)
% pcolor(z,y,real(reshape(result_V_svd{1,1}(N+1:2*N),Ny,Nz)))
% colormap(jet)
% shading interp
% title('V forcing mode')
% hold on
%
% figure(8)
% pcolor(z,y,real(reshape(result_V_svd{1,1}(2*N+1:3*N),Ny,Nz)))
% colormap(jet)
% shading interp
% title('W forcing mode')
% hold on


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