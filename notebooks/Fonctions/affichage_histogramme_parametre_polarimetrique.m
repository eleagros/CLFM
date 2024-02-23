function [val_moyenne, ecart_type] = affichage_histogramme_parametre_polarimetrique(Image, parametre_polarimetrique, limites_pas_histogramme, couleur, position_text_mean, position_text_sigma)

    %% Construction de l'histogramme

    vec = Image(:);
    vec = vec((vec > limites_pas_histogramme(1)) & (vec < limites_pas_histogramme(3)));
    [counts, bins] = hist(vec, [limites_pas_histogramme(1):limites_pas_histogramme(2):limites_pas_histogramme(3)], 'FaceColor', 'blue');

    %% Affichage de l'histogramme

    plot(bins, counts ./ max(counts), 'Linewidth', 1, 'Color', couleur);
    val_moyenne = mean(vec);
    ecart_type = std(vec);
    str1 = strcat('\mu=', num2str(round2(val_moyenne, 0.001)));
    text(position_text_mean(1), position_text_mean(2), str1)
    str2 = strcat('\sigma=', num2str(round2(ecart_type, 0.001)));
    text(position_text_sigma(1), position_text_sigma(2), str2)
    xlim([limites_pas_histogramme(1) limites_pas_histogramme(3)]);
    ylim([0 1.5]);
    grid on;
    grid minor;
    xlabel(parametre_polarimetrique);
    ylabel('Nombre de pixels normalisé');
    title(parametre_polarimetrique, 'FontSize', 15, 'FontName', 'Calibri');

end