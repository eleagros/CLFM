%% Fonction de calibration d'un imageur polarimètrique de Mueller (J. Vizet, le 2017-09-27)
function result_values = calibration_pixels(name_calib, lambda_string, current_folder_calib, flag_transposition_matrices_intensite, flag_calcul_vrai_psi, B0I, P0I, P90I, L30I, Orientation_polariseur, Orientation_lame, SizeL, SizeC)

result_values = [];

% Initialisation des tableaux pour la boucle parfor
sizetotal=SizeL*SizeC;
tabY=linspace(1,SizeC,SizeC);
tabT(1,:)=linspace(1,sizetotal,sizetotal);
tabT(2,:)=reshape(repmat((1:SizeL)',1,SizeC)',sizetotal,1)';
tabT(3,:)=repmat(tabY,1,SizeL);

%% ================== Boucle parfor permettant de calculer les matrices W et A sur un grand nombre de pixels  ======================= %%

parfor ind_boucle=1:1:sizetotal
    
    % Indices permettant de balayer les lignes et les colonnes de l'image
    indL=tabT(2,ind_boucle);
    indC=tabT(3,ind_boucle);
    
    % Transformation des vecteurs 1 x 1 x 16 en matrices d'intensité 4 x 4
    MATRIXB0=reshape(B0I(indL,indC,:),4,4);
    MATRIXPP0=reshape(P0I(indL,indC,:),4,4);
    MATRIXPP90=reshape(P90I(indL,indC,:),4,4);
    MATRIXPL30=reshape(L30I(indL,indC,:),4,4);
     
    if flag_transposition_matrices_intensite==1 % (utile pour les anciennes versions des programmes) 
        MATRIXB0=transpose(MATRIXB0);
        MATRIXPL30=transpose(MATRIXPL30);
        MATRIXPP0=transpose(MATRIXPP0);
        MATRIXPP90=transpose(MATRIXPP90);
    end
    
    %% ================== Algorithme de calibration  ======================= %%
    
    C_elements=MATRIXB0\[MATRIXPP0,MATRIXPP90,MATRIXPL30];
     
    % Protection en cas de pixels abberants
    if any(isnan(C_elements(:)))==1 || any(isinf(C_elements(:)))==1 
        Wtemp=eye(4);
        Atemp=eye(4);  
    else
        
        % Extraction des paramètres Tau, psi, et delta à partir des valeurs propres de Ci
        C_P0=C_elements(1:4,1:4);
        C_P90=C_elements(1:4,5:8);
        C_L30=C_elements(1:4,9:12);
        
        D1=eig(C_P0);
        D2=eig(C_P90);
        D3=eig(C_L30);
        
        % Détermination du tau et du Psi du polariseur à 0°
        D1=abs(D1);
        if D1(:)==zeros(4,1)
            tauP0=0.3;
            psiP0=90;
        else
            tauP0=0.5*(real(max(D1(D1>0))));
            if flag_calcul_vrai_psi==0    
                psiP0=90;   
            else
                psiP0=atan(sqrt(max(D1((real(D1))>0))/min(D1((real(D1))>0))))*180/pi;   
            end
        end
        
        % Détermination du tau et du Psi du polariseur à 90°
        D2=abs(D2);
        if D2(:)==zeros(4,1)
            tauP90=0.3;
            psiP90=0;
        else
            tauP90=0.5*(real(max(D2)));      
            if flag_calcul_vrai_psi==0      
                psiP90=0;   
            else   
                psiP90=atan(sqrt(min(D2((real(D2))>0))/max(D2((real(D2))>0))))*180/pi;    
            end
        end
        
        % Détermination du tau, du Psi et du delta de la lame d'onde
        [~, idx] = sort(imag(D3));
        D3 = D3(idx);
        taulame30=0.5*(real(D3(2))+real(D3(3)));
        psilame30=atand(real(sqrt(real(D3(2))/real(D3(3)))));
        deltalame30 = atan2(imag(D3(4)),real(D3(4)))*180/pi;
        
        % Détermination de la matrice W
        MpolariseurP0=dephaseur_diattenuateur(tauP0,psiP0,0,0);
        MpolariseurP90=dephaseur_diattenuateur(tauP90,psiP90,0,Orientation_polariseur-90);
        Mlame30=dephaseur_diattenuateur(taulame30,psilame30,deltalame30,Orientation_lame);
        
        KRON1=zeros(16,16);
        KRON1(1:4,1:4)=MpolariseurP0;
        KRON1(5:8,5:8)=MpolariseurP0;
        KRON1(9:12,9:12)=MpolariseurP0;
        KRON1(13:16,13:16)=MpolariseurP0;
        
        KRON2=zeros(16,16);
        KRON2(1:4,1:4)=MpolariseurP90;
        KRON2(5:8,5:8)=MpolariseurP90;
        KRON2(9:12,9:12)=MpolariseurP90;
        KRON2(13:16,13:16)=MpolariseurP90;
        
        KRON3=zeros(16,16);
        KRON3(1:4,1:4)=Mlame30;
        KRON3(5:8,5:8)=Mlame30;
        KRON3(9:12,9:12)=Mlame30;
        KRON3(13:16,13:16)=Mlame30;
        
        H1=KRON1-kron(C_P0',eye(4));
        H2=KRON2-kron(C_P90',eye(4));
        H3=KRON3-kron(C_L30',eye(4));
        
        % Méthode compain
        H1S=transpose(H1)*H1;
        H2S=transpose(H2)*H2;
        H3S=transpose(H3)*H3;
        K=H1S+H2S+H3S;
        
        % Méthode Paola
        % K=[H1;H2;H3];
        [~,~,V]=svd(K);
        
        % La matrice W est la dernière colonne de la matrice K (soit le vecteur propre associé à la valeur propre nulle de K)
        Wtemp=reshape(V(:,16),4,4);
        Atemp=MATRIXB0/Wtemp;
        
        % Normalisation des matrices W et A
        Wtemp=Wtemp./Wtemp(1,1);
        Atemp=Atemp./Atemp(1,1);
        
        
    end
    
    % Calcul du conditionnement
    if any(isnan(Wtemp(:)))==1 || any(isinf(Wtemp(:)))==1 || any(isnan(Atemp(:)))==1 || any(isinf(Atemp(:)))==1
        cond_B0(ind_boucle,1)=0;
        cond_W(ind_boucle,1)=0;
        cond_A(ind_boucle,1)=0;
    else
        cond_B0(ind_boucle,1)=1/cond(MATRIXB0);
        cond_W(ind_boucle,1)=1/cond(Wtemp);
        cond_A(ind_boucle,1)=1/cond(Atemp);
    end
    
    % Indexation des matrices W et A sous forme d'images
    if flag_transposition_matrices_intensite==1 
        W(ind_boucle,:)=reshape(Wtemp',1,1,16);
        A(ind_boucle,:)=reshape(Atemp',1,1,16);
    else   
        W(ind_boucle,:)=reshape(Wtemp,1,1,16);
        A(ind_boucle,:)=reshape(Atemp,1,1,16);      
    end  
end

W=reshape(W,SizeC,SizeL,16);
W=permute(W,[2 1 3]);
A=reshape(A,SizeC,SizeL,16);
A=permute(A,[2 1 3]);

% Affichage du conditionnement de W et de A
figure_cond=figure('Position',[311 81 1181 864]);
subplot(3,2,2)
[val_moyenne, ecart_type] = affichage_histogramme_parametre_polarimetrique(cond_W,'\kappa_2^{-1}(W)',[0 0.001 1],'k',[0.1 0.9],[0.1 0.8]);
result_values = [result_values, val_moyenne, ecart_type]; 
subplot(3,2,4)
[val_moyenne, ecart_type] = affichage_histogramme_parametre_polarimetrique(cond_A,'\kappa_2^{-1}(A)',[0 0.001 1],'k',[0.1 0.9],[0.1 0.8]);
result_values = [result_values, val_moyenne, ecart_type]; 
subplot(3,2,6)
[val_moyenne, ecart_type] = affichage_histogramme_parametre_polarimetrique(cond_B0,'\kappa_2^{-1}(B0)',[0 0.001 1],'k',[0.1 0.9],[0.1 0.8]);
result_values = [result_values, val_moyenne, ecart_type];

cond_W=(reshape(cond_W,SizeC,SizeL))';
cond_A=(reshape(cond_A,SizeC,SizeL))';
cond_B0=(reshape(cond_B0,SizeC,SizeL))';

subplot(3,2,1)
imagesc(cond_W);
axis image;
colorbar
h = colorbar;
ylabel(h, '\kappa_2^{-1}(W)')
colormap(jet)

subplot(3,2,3)
imagesc(cond_A);
axis image;
colorbar
h = colorbar;
ylabel(h, '\kappa_2^{-1}(A)')
colormap(jet)

subplot(3,2,5)
imagesc(cond_B0);
axis image;
colorbar
h = colorbar;
ylabel(h, '\kappa_2^{-1}(B0)')
colormap(jet)

% Sauvegarde de la figure montrant les conditionnements de W et de A
supertitle({['Calibration : ' num2str(name_calib) '\' lambda_string 'nm'],''},'FontSize',15,'FontName','Calibri','Interpreter','none');
savefig(gcf,fullfile(current_folder_calib,'Conditionnement'));
%savefig
% Enregistrement des matrices W et A
s2=['\' lambda_string '_W'];
s3=['\' lambda_string '_A'];
sW=strcat(current_folder_calib,s2);
sA=strcat(current_folder_calib,s3);
save(sW,'W');
save(sA,'A');

% Enregistrement des images des conditionnements
s4=['\' lambda_string '_cond_W'];
s5=['\' lambda_string '_cond_A'];
scondW=strcat(current_folder_calib,s4);
scondA=strcat(current_folder_calib,s5);
save(scondW,'cond_W');
save(scondA,'cond_A');

end

