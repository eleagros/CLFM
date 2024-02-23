function[Image_MD,Image_MR,Image_MRL,Image_MRC,Image_Mdelta,Image_total_diattenuation,Image_linear_diattenuation,Image_orientation_linear_diattenuation,Image_circular_diattenuation,Image_total_retardance,Image_retardance_vector,Image_linear_retardance,Image_orientation_linear_retardance,Image_orientation_linear_retardance_full,Image_circular_retardance,Image_total_depolarization]=polar_decomposition_Mueller_image(MM,decomposition_type)
%% === Descriptif ===
% polar_decomposition_Mueller_image : permet de décomposer une image de
% Mueller MM par la décomposition polaire de Lu et Chipman

%% === Commentaires ===


%% === Entrées et sorties ===
% Inputs:
%    MM - Matrice de Mueller sous forme d'image

% Outputs:
%    Image_MD - Image de la Matrice de diatténuation
%    Image_MR - Image de la Matrice du déphaseur
%    Image_MRL - Image de la Matrice du déphaseur linéaire extrait à partir de MR
%    Image_MRC - Image de la Matrice du déphaseur circulaire extrait à partir de MR
%    Image_Mdelta - Image de la Matrice du dépolariseur
%    Image_total_diattenuation - Image de la Diatténuation totale
%    Image_linear_diattenuation - Image de la Diatténuation linéaire
%    Image_orientation linear_diattenuation - Image de l'orientation des axes propres de la
%    Image_diatténuation linéaire
%    Image_circular_diattenuation - Image de la Diatténuation circulaire
%    Image_total_retardance - Image du Retard de phase total (correspond au RetS) extrait
%    à partir de MR
%    Image_retardance_vector - Image du vecteur de retard de phase extrait à partir de MR
%    Image_linear_retardance - Image du Retard de phase linéaire extrait à partir de MRL
%    Image_orientation_linear_retardance - Image de l'orientation des axes propres du retard
%    de phase linéaire (donné entre 0° et 90°)
%    Image_orientation_linear_retardance_full - Image de l'Orientation des axes propres du 
%    retard de phase linéaire (donné entre 0° et 180°). Vrai à condition
%    que le retard de phase linéaire n'excède pas 180°
%    Image_circular_retardance - Image du Retard de phase circulaire
%    Image_total_depolarization - Image de la Dépolarisation totale calculée à partir de
%    Mdelta

%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% December 2017; Last revision: 2017-02-17



%% ------------- BEGIN CODE --------------

[sizeL,sizeC,sizeD]=size(MM);

%% =================  Initialisation des tableaux  ================= %%


Image_MD=zeros(sizeL,sizeC,sizeD);
Image_MR=zeros(sizeL,sizeC,sizeD);
Image_Mdelta=zeros(sizeL,sizeC,sizeD);
Image_MRL=zeros(sizeL,sizeC,sizeD);
Image_MRC=zeros(sizeL,sizeC,sizeD);

Image_total_diattenuation=zeros(sizeL,sizeC);
Image_linear_diattenuation=zeros(sizeL,sizeC);
Image_orientation_linear_diattenuation=zeros(sizeL,sizeC);
Image_circular_diattenuation=zeros(sizeL,sizeC);

Image_total_retardance=zeros(sizeL,sizeC);
Image_retardance_vector= repmat(reshape(zeros(1,4),1,1,4),sizeL,sizeC);
Image_linear_retardance=zeros(sizeL,sizeC);
Image_orientation_linear_retardance=zeros(sizeL,sizeC);
Image_orientation_linear_retardance_full=zeros(sizeL,sizeC);
Image_circular_retardance=zeros(sizeL,sizeC);

Image_total_depolarization=zeros(sizeL,sizeC);


h = waitbar(0,'Please wait...');


%% =================  Décomposition de l'image de Mueller  ================= %%

for indL=1:sizeL
    
    for indC=1:sizeC
        
        muel=(reshape(MM(indL,indC,:),4,4));
        muel(1,1)=1;
        
        %% Lancement de la décomposition de la matrice de Mueller M
        
        [MD,MR,MRL,MRC,Mdelta,total_diattenuation,linear_diattenuation,orientation_linear_diattenuation,circular_diattenuation,total_retardance,retardance_vector,linear_retardance,orientation_linear_retardance,orientation_linear_retardance_full,circular_retardance,total_depolarization]=polar_decomposition_Mueller(muel,decomposition_type);
        
        
        %% Images des matrices MD (diatténuateur), MR (déphaseur), Mdelta (dépolariseur) %%
        
        Image_MD(indL,indC,:)=real(reshape(MD,1,1,16));
        Image_MR(indL,indC,:)=reshape(MR,1,1,16);
        Image_MRL(indL,indC,:)=reshape(MRL,1,1,16);
        Image_MRC(indL,indC,:)=reshape(MRC,1,1,16);
        Image_Mdelta(indL,indC,:)=reshape(Mdelta,1,1,16);
        
        %% Images des paramètres de la diatténuation %%
        
        Image_total_diattenuation(indL,indC)=total_diattenuation;
        Image_linear_diattenuation(indL,indC)=linear_diattenuation;
        Image_orientation_linear_diattenuation(indL,indC)=orientation_linear_diattenuation;
        Image_circular_diattenuation(indL,indC)=circular_diattenuation;
        
        %% Images des paramètres du retard de phase %%
        
        Image_total_retardance(indL,indC)=total_retardance;
        Image_retardance_vector(indL,indC,:)=retardance_vector;
        Image_linear_retardance(indL,indC)=linear_retardance;
        Image_orientation_linear_retardance(indL,indC)=orientation_linear_retardance;
        Image_orientation_linear_retardance_full(indL,indC)=orientation_linear_retardance_full;
        Image_circular_retardance(indL,indC)=circular_retardance;
        
        %% Images du paramètre de dépolarisation totale %%
        
        Image_total_depolarization(indL,indC)=total_depolarization;
        
    end
    
    waitbar(indL/sizeL);
end

close(h);

end

%% ------------- END OF CODE --------------
