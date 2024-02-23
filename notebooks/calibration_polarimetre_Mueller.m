%% Calibration polarimètre de Mueller (J. Vizet, le 2017-09-27)
format long;
% clear
close all
clear all

addpath('Fonctions'); 
disp('here')
%close all

%% ============== Paramétrage du programme de calibration ======================= %%

CalibrationAvecLabview=0;                   % Si le programme est destiné à être lancé à partir de LabVIEW, mettre ce paramètre à 1
CalibrationImageur=1;                       % Si le programme doit calibrer un imageur, mettre ce paramètre à 1. Si CalibrationAvecLabview=0, il faut forcément mettre CalibrationImageur=1.
flag_transposition_matrices_intensite=0;    % Si ce paramètre est à 1, les matrices intensités sont transposées avant de lancer l'algorithme de calibration
flag_calcul_vrai_psi=0;                     % Si ce paramètre est à 1, les valeurs des vrais "psi" (angles de dichroïsmes) du polariseur à 0° et 90° sont calculées au lieu d'être imposées dans le programme.



current_folder_calib = textread('tmp/current_folder_calib.txt', '%s', 'delimiter', '\n');
current_folder_calib = current_folder_calib{1};
 
if CalibrationImageur==1
    [TailleImage,GammeDynamique]=parametrage_camera('Stingray IPM1')
    SizeL=TailleImage(1);
    SizeC=TailleImage(2);
end


% current_folder_calib = 'C:\Users\romai\Documents\Zoom\calib';
%% =================== Importation des fichiers d'intensité ===================== %%

[path_rest,lambda_complet,~]=fileparts(current_folder_calib);       % Exemple de lambda_complet : 550nm
[~,name_calib,~]=fileparts(path_rest);                              % Exemple de name_calib : 2015-02-03_D_2
lambda_string=lambda_complet(1:end-2);                              % Exemple de lambda_string : 550


% Calibration dans le cas d'un imageur polarimétrique de Mueller
if CalibrationImageur==1
    
    [B0I,sizeC,sizeL,exposuretime]=readCOD(fullfile(current_folder_calib,[lambda_string '_B0.cod']));
    [P0I,sizeC,sizeL,exposuretime]=readCOD(fullfile(current_folder_calib,[lambda_string '_P0.cod']));
    [P90I,sizeC,sizeL,exposuretime]=readCOD(fullfile(current_folder_calib,[lambda_string '_P90.cod']));
    [L30I,sizeC,sizeL,exposuretime]=readCOD(fullfile(current_folder_calib,[lambda_string '_L30.cod']));
    [BruitI,sizeC,sizeL,exposuretime]=readCOD(fullfile(current_folder_calib,[lambda_string '_Bruit.cod']));
    
    B0I=B0I-BruitI;
    P0I=P0I-BruitI;
    P90I=P90I-BruitI;
    L30I=L30I-BruitI;
    
    % Définition de la zone des images à moyenner : par définition, un carré de 100 pixels x 100 pixels au centre de l'image
    point_central=[ceil(SizeL/2),ceil(SizeC/2)];                                         
    LineStart=point_central(1)-50;
    LineEnd=point_central(1)+49;
    ColStart=point_central(2)-50;
    ColEnd=point_central(2)+49;
   
    % Moyenne des images d'intensité
    for j=1:1:16                                                                         
        B0(1,1,j)=mean(mean(B0I(LineStart:LineEnd,ColStart:ColEnd,j)))  ;
        P0(1,1,j)=mean(mean(P0I(LineStart:LineEnd,ColStart:ColEnd,j)))  ;
        P90(1,1,j)=mean(mean(P90I(LineStart:LineEnd,ColStart:ColEnd,j)))  ;
        L30(1,1,j)=mean(mean(L30I(LineStart:LineEnd,ColStart:ColEnd,j)))  ;
        Bruit(1,1,j)=mean(mean(BruitI(LineStart:LineEnd,ColStart:ColEnd,j)))  ;
    end
    
    % Transformation des vecteurs 1 x 1 x 16 en matrices d'intensité 4 x 4
    MATRIXB0=reshape(B0,4,4);
    MATRIXPP0=reshape(P0,4,4);
    MATRIXPP90=reshape(P90,4,4);
    MATRIXPL30=reshape(L30,4,4);

% Calibration dans le cas d'un polarimètre avec un seul point de mesure    
else
    
    PB0=csvread([current_folder_calib '\' 'B0.csv']);                                    
    PP0=csvread([current_folder_calib '\' 'P0.csv']);
    PP90=csvread([current_folder_calib '\' 'P90.csv']);
    PL30=csvread([current_folder_calib '\' 'L30.csv']);
    
    if (exist(fullfile([current_folder_calib '\'], 'Bruit.csv')))==2
        PBRUIT=csvread([current_folder_calib '\' 'Bruit.csv']);
        Bruit=moyennage_csv(PBRUIT);
    else
        Bruit=zeros(4,4);
    end
    
    % Transformation des vecteurs 1 x 1 x 16 en matrices d'intensité 4 x 4
    MATRIXB0=moyennage_csv(PB0)-Bruit;
    MATRIXPP0=moyennage_csv(PP0)-Bruit;
    MATRIXPP90=moyennage_csv(PP90)-Bruit;
    MATRIXPL30=moyennage_csv(PL30)-Bruit;
    
end

% (utile pour les anciennes versions des programmes)
if flag_transposition_matrices_intensite==1                                               
    MATRIXB0=transpose(MATRIXB0);
    MATRIXPL30=transpose(MATRIXPL30);
    MATRIXPP0=transpose(MATRIXPP0);
    MATRIXPP90=transpose(MATRIXPP90);  c
end


%% ================== Algorithme de calibration (1ère partie) ======================= %%

% Extraction des paramètres Tau, psi, et delta à partir des valeurs propres de Ci
C_elements=MATRIXB0\[MATRIXPP0,MATRIXPP90,MATRIXPL30];
C_P0=C_elements(1:4,1:4);
C_P90=C_elements(1:4,5:8);
C_L30=C_elements(1:4,9:12);

D1=eig(C_P0);                                                                             
D2=eig(C_P90);
D3=eig(C_L30);

 % Détermination du tau et du Psi du polariseur à 0°
D1=abs(D1);                                                                              
tauP0=0.5*(real(max(D1(D1>0))));
if flag_calcul_vrai_psi==0 
    psiP0=90; 
else
    psiP0=atan(sqrt(max(D1((real(D1))>0))/min(D1((real(D1))>0))))*180/pi;  
end

% Détermination du tau et du Psi du polariseur à 90°
D2=abs(D2);
tauP90=0.5*(real(max(D2)));
if flag_calcul_vrai_psi==0
    psiP90=0;
else  
    psiP90=atan(sqrt(min(D2((real(D2))>0))/max(D2((real(D2))>0))))*180/pi;    
end

% Détermination du tau, du Psi et du delta de la lame d'onde
[~,idx] = sort(imag(D3));
D3 = D3(idx);
taulame30=0.5*(real(D3(2))+real(D3(3)));
psilame30=atand(real(sqrt(real(D3(2))/real(D3(3)))));
deltalame30 = atan2(imag(D3(4)), real(D3(4)))*180/pi;


%% ================== Algorithme de calibration (2ème partie) ======================= %%

pas_orientation=0.1;  % Pas de calcul pour les orientations du polariseur et de la lame de phase
vec_orientation_polariseur=-10:pas_orientation:10;  % Vecteur contenant toutes les orientations du polariseur qui seront testées dans la boucle
vec_orientation_lame=20:pas_orientation:40;    % Vecteur contenant toutes les orientations de la lame qui seront testées dans la boucle
total_points=length(vec_orientation_polariseur)*length(vec_orientation_lame);  % Nombre total de points calculés


% Initialisation des variables de la boucle d'optimisation
tab_optimisation=zeros(3,total_points);
ind_boucle=0;
compteur=0;
MpolariseurP0=dephaseur_diattenuateur(tauP0,psiP0,0,0);

% Vectorisation
H1=kron(eye(4),MpolariseurP0)-kron(C_P0',eye(4));
H1S=transpose(H1)*H1;
h = waitbar(0,'Please wait...');

% Boucle d'optimisation de l'ECM
for ind_boucle_polariseur=vec_orientation_polariseur
    
    compteur=compteur+1;
    % Construction des Mi à partir des valeurs propres de Ci
    MpolariseurP90=dephaseur_diattenuateur(tauP90,psiP90,0,ind_boucle_polariseur);
    % Vectorisation
    H2=kron(eye(4),MpolariseurP90)-kron(C_P90',eye(4));
      H2S=transpose(H2)*H2;
      
       
    for ind_boucle_lame=vec_orientation_lame
        
        ind_boucle=ind_boucle+1;
        
        % Construction des Mi à partir des valeurs propres de Ci
        Mlame30=dephaseur_diattenuateur(taulame30,psilame30,deltalame30,ind_boucle_lame);
        
        % Vectorisation
        H3=kron(eye(4),Mlame30)-kron(C_L30',eye(4));
        
        % Méthode Compain 
        H3S=transpose(H3)*H3;
        K=H1S+H2S+H3S;
        
        [S]=svd(K);
        
        tab_optimisation(1,ind_boucle)=ind_boucle_polariseur;
        tab_optimisation(2,ind_boucle)=ind_boucle_lame;
        tab_optimisation(3,ind_boucle)=10*log10(S(16)/S(15));
        
    end
    
    waitbar(compteur/length(vec_orientation_polariseur))
    
end

close(h);

% Détection du minimum du rapport des deux dernières valeurs propres de la matrice K
tab_optimisation(1,:)=tab_optimisation(1,:)+90;
Z=min(tab_optimisation(3,:));
index_min=find(tab_optimisation(3,:)==Z);

% Orientation du polariseur à 90° et de la lame à 30° après la procédure de calibration
Orientation_polariseur=tab_optimisation(1,index_min);
Orientation_lame=tab_optimisation(2,index_min);

%% ================== Affichage de la courbe du rapport lambda16/lambda 15 en fonction de l'angle du polariseur et de la lame d'onde ======================= %%

f=figure
[x,y]=meshgrid(vec_orientation_polariseur+90,vec_orientation_lame);
z = reshape(tab_optimisation(3,:),length(vec_orientation_lame),length(vec_orientation_polariseur));
surfc(x,y,z,'FaceColor','interp','EdgeColor','none','FaceLighting','phong')
grid on;
xlabel('Orientation du P90 (°)');
ylabel('Orientation de la L30 (°)');
zlabel('Ratio \lambda 16 / \lambda 15 (en dB)');
axis([80 100 20 40]);
% axis([60 120 0 60]);
axis square;
colormap(jet)
view(-45,20)
title({['Calibration : ' num2str(name_calib) '\' lambda_string 'nm'],''},'interpreter','none')
saveFigure(gcf,fullfile(current_folder_calib,'Solution de la calibration'));
%savefig(gcf,fullfile(current_folder_calib,'Solution de la calibration'));
% print(f,fullfile(current_folder_calib,'Solution de la calibration'),'-dpng');


%% ================== Choix de calibration (imageur ou un seul point) ======================= %%

if CalibrationImageur==1
    
    % Sauvegarde des paramètres de la calibration dans un fichier texte
    Parametres_calibration = {'O_L30 (degrés)';'O_P90 (degrés)';'T_P0 (normalisé)';'T_P90 (normalisé)';'T_L30 (normalisé)';'Psi_P0 (degrés)';'Psi_P90 (degrés)';'Psi_L30 (degrés)';'R_L30 (degrés)';'Ratio VP (dB)'};
    Valeurs_Parametres_Calibration = [Orientation_lame;Orientation_polariseur;tauP0;tauP90;taulame30;psiP0;psiP90;psilame30;deltalame30;Z];
    Text_file=concatParam(Parametres_calibration,Valeurs_Parametres_Calibration);
    fid=fopen(fullfile(current_folder_calib,'Logbook_calibration.txt'),'wt');
    fprintf(fid,Text_file(3:end));
    fclose(fid);

    % Lancement de la calibration pixel par pixel
    val = calibration_pixels(name_calib,lambda_string,current_folder_calib,flag_transposition_matrices_intensite,flag_calcul_vrai_psi,B0I,P0I,P90I,L30I,Orientation_polariseur,Orientation_lame,SizeL,SizeC);
    save(fullfile(current_folder_calib,'values.mat'), 'val');

else
    
    % Détermination de la matrice W
    MpolariseurP0=dephaseur_diattenuateur(tauP0,psiP0,0,0);
    MpolariseurP90=dephaseur_diattenuateur(tauP90,psiP90,0,Orientation_polariseur-90);
    Mlame30=dephaseur_diattenuateur(taulame30,psilame30,deltalame30,Orientation_lame);
    
    H1=kron(eye(4),MpolariseurP0)-kron(C_P0',eye(4));
    H2=kron(eye(4),MpolariseurP90)-kron(C_P90',eye(4));
    H3=kron(eye(4),Mlame30)-kron(C_L30',eye(4));
    
    % Méthode compain  
    H1S=transpose(H1)*H1;
    H2S=transpose(H2)*H2;
    H3S=transpose(H3)*H3;
    K=H1S+H2S+H3S;
    
    [~,S,V]=svd(K);
    
    % La matrice W est la dernière colonne de la matrice K (soit le vecteur propre associé à la valeur propre nulle de K)
    W=reshape(V(:,16),4,4);
    
    % Calcul de A à partir de B0 et de W
    A=MATRIXB0*inv(W);
    
    % Normalisation des matrices W et A   
    W=W./W(1,1)
    A=A./A(1,1)
    
    % Inverses des conditionnements de W et de A
    conditionnement_W=1/cond(W);
    conditionnement_A=1/cond(A);
    
    % Sauvegarde des paramètres de la calibration dans un fichier texte
    Parametres_calibration = {'O_L30 (degrés)';'O_P90 (degrés)';'T_P0 (normalisé)';'T_P90 (normalisé)';'T_L30 (normalisé)';'Psi_P0 (degrés)';'Psi_P90 (degrés)';'Psi_L30 (degrés)';'R_L30 (degrés)';'Ratio VP (dB)';'Inv. du conditionnement de W';'Inv. du conditionnement de A'};
    Valeurs_Parametres_Calibration = [Orientation_lame;Orientation_polariseur;tauP0;tauP90;taulame30;psiP0;psiP90;psilame30;deltalame30;Z;conditionnement_W;conditionnement_A];
    Text_file=concatParam(Parametres_calibration,Valeurs_Parametres_Calibration);
    fid=fopen(fullfile(current_folder_calib,'Logbook_calibration.txt'),'wt');
    fprintf(fid,Text_file(3:end));
    fclose(fid);
    
    % Sauvegarde de W et de A 
    s2=['\' lambda_string '_W'];
    s3=['\' lambda_string '_A'];
    
    sW=strcat(current_folder_calib,s2);
    sA=strcat(current_folder_calib,s3);
    
    save(sW,'W');
    save(sA,'A');
    
end




