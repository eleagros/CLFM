function [MM,M11] = CalculMM(A,W,INTENSITE,SizeIM,option_transposition)

%% Jérémy Vizet, le 2017-09-27 : Calcul de la matrice de Mueller à partir des matrices de calibration W et A
 
%% Préallocation de la mémoire

M11=zeros(SizeIM(4),SizeIM(3));
MM=zeros(SizeIM(4),SizeIM(3),16);


for indL=1:SizeIM(4)
    for indC=1:SizeIM(3)
        
        Atemp = reshape(A(SizeIM(2)+indL-1,SizeIM(1)+indC-1,:),4,4);
        Wtemp = reshape(W(SizeIM(2)+indL-1,SizeIM(1)+indC-1,:),4,4);
        Itemp = reshape(INTENSITE(SizeIM(2)+indL-1,SizeIM(1)+indC-1,:),4,4);
        
        if option_transposition==1
            MMtemp = inv(Atemp')*(Itemp'*inv(Wtemp'));
            MMtemp=MMtemp';
            
        else
            MMtemp=Atemp\Itemp/Wtemp;
            
        end
        
        M11_temp = MMtemp(1,1);
        MMtemp_norm=MMtemp/M11_temp;
        M11(indL,indC) = M11_temp;
        MM(indL,indC,:)=MMtemp_norm(:);
    end
end

end