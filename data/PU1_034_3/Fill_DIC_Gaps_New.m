clc;
clear all;
close all;

addpath 'C:\Users\cl46676\Desktop\Pressure Ulcers\ncorr_2D_matlab-master'

OLD_structure = 'PU1_034_3.mat';
FIX_structure = ['NEWLYFIXED_','PU1_034_3.mat'];
GIFF_folderPath  = './GIFFs_NewlyFixed/';
load('PU1_034_3.mat')

fig_size = [5,4]; %figure size in inches
n_img = length(current_save);
if ~exist(GIFF_folderPath, 'dir')
   mkdir(GIFF_folderPath)
end

fprintf('Exx_cur \n')
for j = 1:n_img 
    I = data_dic_save.strains(j).plot_exx_cur_formatted;
    I(find(I==0))= nan;
    F = fillmissing2(I,'cubic');
    F(find(isnan(F)))=0;
    data_dic_save.strains(j).plot_exx_cur_formatted = F;   
    Exx_cur{j} = F;
    
    %Shear Strain
    I = data_dic_save.strains(j).plot_exy_cur_formatted;
    I(find(I==0))=nan;
    F = fillmissing2(I,'cubic');
    F(find(isnan(F)))=0;
    Exy_cur{j} = F;
    % Find The Boundaries and fill the mask
    % Correct mask
    M = data_dic_save.strains(j).roi_cur_formatted.mask;
%     BW = edge(M,'canny',[0, 0.2]);
%     
%     %Scan image rows
%     Mnew = M;
%     for jj = 1:size(M,1)
%         min_bound = min(find(BW(jj,:)==1))+px_margin;
%         max_bound = max(find(BW(jj,:)==1))-px_margin;
%         temp = M(jj,min_bound:max_bound);
%         points_found = length(find(temp==0));
%         if points_found<max_gap
%             temp(find(temp==0)) = 1;
%         end
%         Mnew(jj,min_bound:max_bound) = temp;
%     end
    
    Mnew  = imfill( M ,'holes');
    Mask_cur{j} = Mnew; 
    data_dic_save.strains(j).roi_cur_formatted.mask = Mnew;
end

fprintf('Exx_ref \n')
for j = 1:n_img
    I = data_dic_save.strains(j).plot_exx_ref_formatted;
    I(find(I==0))=nan;
    F = fillmissing2(I,'cubic');
    F(find(isnan(F)))=0;
    data_dic_save.strains(j).plot_exx_ref_formatted = F;
    Exx_ref{j} = F;
    stretch_x{j} = sqrt(2*F+1);
    
    %Shear Strain
    I = data_dic_save.strains(j).plot_exy_ref_formatted;
    I(find(I==0))=nan;
    F = fillmissing2(I,'cubic');
    F(find(isnan(F)))=0;
    Exy_ref{j} = F;
    
%     % Find The Boundaries and fill the mask
%     M = data_dic_save.strains(j).roi_ref_formatted.mask;
%     BW = edge(M,'canny',[0, 0.99]);
%     BW2 = imfill(BW,'holes');
%     data_dic_save.strains(j).roi_ref_formatted.mask = BW2; 
    M = data_dic_save.strains(j).roi_ref_formatted.mask;
%     BW = edge(M,'canny');
%     %Scan image rows
%     Mnew = M;
%     for jj = 1:size(M,1)
%         min_bound = min(find(BW(jj,:)==1))+px_margin;
%         max_bound = max(find(BW(jj,:)==1))-px_margin;
%         temp = M(jj,min_bound:max_bound);
%         points_found = length(find(temp==0));
%         if points_found<max_gap
%             temp(find(temp==0)) = 1;
%         end
%         Mnew(jj,min_bound:max_bound) = temp;
%     end

    Mnew  = imfill( M ,'holes');
    Mask_ref{j} = Mnew; 
    data_dic_save.strains(j).roi_ref_formatted.mask = Mnew;
     
end

fprintf('Eyy_cur \n')
for j = 1:n_img
    I = data_dic_save.strains(j).plot_eyy_cur_formatted;
    I(find(I==0))=nan;
    F = fillmissing2(I,'cubic');
    F(find(isnan(F)))=0;
    data_dic_save.strains(j).plot_eyy_cur_formatted = F;
    Eyy_cur{j} = F;
end

fprintf('Eyy_ref \n')
for j = 1:n_img
    I = data_dic_save.strains(j).plot_eyy_ref_formatted;
    I(find(I==0))=nan;
    F = fillmissing2(I,'cubic');
    F(find(isnan(F)))=0;
    data_dic_save.strains(j).plot_eyy_ref_formatted = F;
    Eyy_ref{j} = F;
    stretch_y{j} = sqrt(2*F+1);
end


% Caclulate max principal strain
% REFERENCE
fprintf('Max principal strain (it may take a while) \n')
for j = 1:n_img
    
    % REF 
    Exx = Exx_ref{j};
    Exy = Exy_ref{j};
    Eyy = Eyy_ref{j};
    
    row_pix = size(Exx,1);
    col_pix = size(Exx,2);
    MaxP = [];
    for jj= 1:row_pix
        for kk = 1:col_pix
            
            Etemp = [Exx(jj,kk),Exy(jj,kk);Exy(jj,kk),Eyy(jj,kk)];
            temp = eig(Etemp);
            MaxP(jj,kk) = max(temp);
            
        end
    end
    MaxP_ref{j} = MaxP;
    
    % CURRENT 
    Exx = Exx_cur{j};
    Exy = Exy_cur{j};
    Eyy = Eyy_cur{j};
    
    row_pix = size(Exx,1);
    col_pix = size(Exx,2);
    MaxP = [];
    for jj= 1:row_pix
        for kk = 1:col_pix
            
            Etemp = [Exx(jj,kk),Exy(jj,kk);Exy(jj,kk),Eyy(jj,kk)];
            temp = eig(Etemp);
            MaxP(jj,kk) = max(temp);
            
        end
    end
    MaxP_cur{j} = MaxP;
            
end

save(FIX_structure,"current_save","reference_save","data_dic_save");


%% Create My custom color map
vec = [100;  50;    0];
hex = ['#128fcd';'#DCDCDC';'#cc003d'];
raw = sscanf(hex','#%2x%2x%2x',[3,size(hex,1)]).' / 255;
N = 128;
%N = size(get(gcf,'colormap'),1) % size of the current colormap
% map = interp1(vec,raw,linspace(100,0,N),'pchip');
map = interp1(vec,raw,linspace(100,0,N));


%% Plot MAX PRINCIPAL STRAIN REF
h = figure('units','inch','position',[2,2,fig_size(1),fig_size(2)]);
axis tight manual % this ensures that getframe() returns a consistent size
filename = [GIFF_folderPath,'MaxP_ref.gif'];
% temp = cell2mat(MaxP_ref);
% Cbounds = [min(min(temp)), max(max(temp))];
Cbounds = [0,2]; % The range was set by Christina. Can be adjusted if needed.
for n = 1:n_img
    % Draw plot for y = x.^n
    data = MaxP_ref{n};
    Msk=Mask_ref{n};
    %Cbounds = [1.00,2.05];
    writegifDIC(h, filename, data, Msk, Cbounds, n, map)
end


%% Plot MAX PRINCIPAL STRAIN CURRENT
h = figure('units','inch','position',[2,2,fig_size(1),fig_size(2)]);
axis tight manual % this ensures that getframe() returns a consistent size
filename = [GIFF_folderPath,'MaxP_cur.gif'];
% temp = cell2mat(MaxP_cur);
% Cbounds = [min(min(temp)), max(max(temp))];
Cbounds = [0,0.5]; % The range was set by Christina. Can be adjusted if needed.
for n = 1:n_img
    % Draw plot for y = x.^n
    data = MaxP_cur{n};
    Msk=Mask_cur{n};
    %Cbounds = [1.00,2.05];
    writegifDIC(h, filename, data, Msk, Cbounds, n, map)
end


%% Plot Eyy Cur
h = figure('units','inch','position',[2,2,fig_size(1),fig_size(2)]);
axis tight manual % this ensures that getframe() returns a consistent size
filename = [GIFF_folderPath,'Eyy_cur.gif'];
% temp = cell2mat(Eyy_cur);
% Cbounds = [min(min(temp)), max(max(temp))];
Cbounds = [0,0.4]; % The range was set by Christina. Can be adjusted if needed.
for n = 1:n_img
    % Draw plot for y = x.^n
    data = Eyy_cur{n};
    Msk=Mask_cur{n};
    %Cbounds = [0.0,0.45];
    writegifDIC(h, filename, data, Msk, Cbounds, n, map)
end


%% Plot Exx Cur
h = figure('units','inch','position',[2,2,fig_size(1),fig_size(2)]);
axis tight manual % this ensures that getframe() returns a consistent size
filename = [GIFF_folderPath,'Exx_cur.gif'];
% temp = cell2mat(Exx_cur);
% Cbounds = [min(min(temp)), max(max(temp))];
Cbounds = [0,0.4]; % The range was set by Christina. Can be adjusted if needed.
for n = 1:n_img
    % Draw plot for y = x.^n
    data = Exx_cur{n};
    Msk=Mask_cur{n};
    %Cbounds = [-0.3,0.02];
    writegifDIC(h, filename, data, Msk, Cbounds, n, map)
end


% %% Plot Eyy REF
% h = figure('units','inch','position',[2,2,fig_size(1),fig_size(2)]);
% axis tight manual % this ensures that getframe() returns a consistent size
% filename = [GIFF_folderPath,'Eyy_ref.gif'];
% temp = cell2mat(Eyy_ref);
% Cbounds = [min(min(temp)), max(max(temp))];
% for n = 1:n_img
%     % Draw plot for y = x.^n
%     data = Eyy_ref{n};
%     Msk=Mask_ref{n};
%     %Cbounds = [-0.2,1.6];
%     writegifDIC(h, filename, data, Msk, Cbounds, n, map)
% end


% %% Plot Exx REF
% h = figure('units','inch','position',[2,2,fig_size(1),fig_size(2)]);
% axis tight manual % this ensures that getframe() returns a consistent size
% filename = [GIFF_folderPath,'Exx_ref.gif'];
% temp = cell2mat(Exx_ref);
% Cbounds = [min(min(temp)), max(max(temp))];
% for n = 1:n_img
%     % Draw plot for y = x.^n
%     data = Exx_ref{n};
%     Msk=Mask_ref{n};
%     %Cbounds = [-0.18,0.1];
%     writegifDIC(h, filename, data, Msk, Cbounds, n, map)
% end


% %% Plot STRETCH yy REF
% h = figure('units','inch','position',[2,2,fig_size(1),fig_size(2)]);
% axis tight manual % this ensures that getframe() returns a consistent size
% filename = [GIFF_folderPath,'StretchYY_ref.gif'];
% temp = cell2mat(stretch_y);
% Cbounds = [min(min(temp)), max(max(temp))];
% for n = 1:n_img
%     % Draw plot for y = x.^n
%     data = stretch_y{n};
%     Msk=Mask_ref{n};
%     %Cbounds = [1.00,2.05];
%     writegifDIC(h, filename, data, Msk, Cbounds, n, map)
% end


% %% Plot STRETCH xx REF
% h = figure('units','inch','position',[2,2,fig_size(1),fig_size(2)]);
% axis tight manual % this ensures that getframe() returns a consistent size
% filename = [GIFF_folderPath,'StretchXX_ref.gif'];
% temp = cell2mat(stretch_x);
% Cbounds = [min(min(temp)), max(max(temp))];
% for n = 1:n_img
%     % Draw plot for y = x.^n
%     data = stretch_x{n};
%     Msk=Mask_ref{n};
%     %Cbounds = [0.85,1.02];
%     writegifDIC(h, filename, data, Msk, Cbounds, n, map)
% end


%% FUNCTIONS

function []=writegifDIC(h, filename, data, Msk, Cbounds, n, map)

    img_data=nan(size(data));
    img_data(Msk==1) = data(Msk==1);
    
    c = imagesc(img_data);
    set(c, 'AlphaData', 1-isnan(img_data))
    colormap(map)
    caxis(Cbounds)
    colorbar
    drawnow
    axis off
    set(gca, 'color', 'w','box','off');
    set(gcf, 'Color', 'w')
%     saveas(h,['./PNGsElla/',filename,'_',num2str(n),'.png'])
    saveas(h,[filename,'_',num2str(n),'.tiff'])
      % Capture the plot as an image 
      frame = getframe(h); 
      im = frame2im(frame); 
      [imind,cm] = rgb2ind(im,256); 
      % Write to the GIF File 
      if n == 1 
          imwrite(imind,cm,filename,'gif', 'Loopcount',inf,'DelayTime',0.04,'DisposalMethod','restoreBG'); 
      else 
          imwrite(imind,cm,filename,'gif','WriteMode','append','DelayTime',0.04,'DisposalMethod','restoreBG'); 
      end 
end


% [nr,nc] = size(A);
% pcolor([A zeros(nr,1); zeros(1,nc+1)]);
%  shading flat;
%  set(gca, 'ydir', 'reverse');