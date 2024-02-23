function affichage_matrice_mueller_old(MM,range_colorbar)
I=0;
gg=0;
for j=1:4
    for k=1:4
        I=I+1;
        gg=(k-1)*4+j; % transposition pour l'affichage
        subplot(4,4,I)
        imagesc(MM(:,:,gg))
        axis image
        colorbar
        colormap(colormap_perso)
        caxis([range_colorbar(1) range_colorbar(2)]);
       
     end
end

end