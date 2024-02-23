function[Image_eigenvalues_coherency,Masks_eigenvalues_coherency]=eigenvalues_coherency_image(MM)

%% === Descriptif ===
%eigenvalues_coherency_image - Permet de donner les images des quatre valeurs propres
% de la matrice de cohérence de la matrice de Mueller sous forme d'image

%% === Commentaires ===


%% === Entrées et sorties ===
% Inputs:
%    MM - Image d'une matrice de Mueller à filtrer
%
% Outputs:
%    Image_eigenvalues_coherency - Image_eigenvalues_coherency
%    (nombre_pixels_ligne,nombre_pixel_colonne,numero_valeur_propre) -
%    Images des valeurs propres de la matrice de cohérence
%    Les valeurs propres sont classées par ordre décroissant
%    Masks_eigenvalues_coherency - Masks_eigenvalues_coherency
%    (nombre_pixels_ligne,nombre_pixel_colonne,numero_valeur_propre) -
%    Masques affichant les valeurs propres négatives sur toutes les images
%    des valeurs propres


%% === Signature ===
% Author: Jeremy Vizet, Ph.D., optical polarimetry
% Ecole polytechnique, LPICM
% email address: jeremy.vizet@polytechnique.edu
% March 2018; Last revision: 2018-03-30

%% ------------- BEGIN CODE --------------

[sizeL,sizeC,sizeD]=size(MM);


h = waitbar(0,'Please wait...');

for indL=1:sizeL
    
    for indC=1:sizeC
        
        muel=(reshape(MM(indL,indC,:),4,4));
        muel(1,1)=1;
        
        [~,~,raw_eigenvalues,~]=physical_crit(muel);
        sorted_eigenvalues = sort(diag(raw_eigenvalues),'descend');
        Image_eigenvalues_coherency(indL,indC,1)=sorted_eigenvalues(1);
        Image_eigenvalues_coherency(indL,indC,2)=sorted_eigenvalues(2);  
        Image_eigenvalues_coherency(indL,indC,3)=sorted_eigenvalues(3);
        Image_eigenvalues_coherency(indL,indC,4)=sorted_eigenvalues(4);
               
     
    end
    
    waitbar(indL/sizeL);
    
end

        Masks_eigenvalues_coherency(:,:,1)=Image_eigenvalues_coherency(:,:,1)>0;
        Masks_eigenvalues_coherency(:,:,2)=Image_eigenvalues_coherency(:,:,2)>0;
        Masks_eigenvalues_coherency(:,:,3)=Image_eigenvalues_coherency(:,:,3)>0;
        Masks_eigenvalues_coherency(:,:,4)=Image_eigenvalues_coherency(:,:,4)>0;
        
        
close(h);

end

%% ------------- END OF CODE --------------