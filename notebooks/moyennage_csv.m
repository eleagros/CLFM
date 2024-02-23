function [ matrice_intensite ] = moyennage_csv(fichier_csv_banc_optique)


ind_boucle=0;
matrice_intensite=zeros(1,16);

for k=1:1:16
    for l=4+ind_boucle:1:7+ind_boucle
        matrice_intensite(1,k)=(matrice_intensite(1,k)+fichier_csv_banc_optique(l,1));
    end
    ind_boucle=ind_boucle+10;
end

matrice_intensite=matrice_intensite/4;
matrice_intensite=reshape(matrice_intensite,4,4);
    
end