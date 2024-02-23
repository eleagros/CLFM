function PackedVar = Packing(varargin)
% Packing fait la somme de la variable sur un carré de Pck x Pck de côté
% il renvoie soit une matrice de même taille que Variable, avec des Nan
% dans la bbande de pixels restant (modulo Pck), soit une matrice de taille
% réduite (multiple de Pck)

Variable = varargin{1};
Pck = varargin{2};

if nargin >= 3
    SameSize = varargin{3};
else
    SameSize = 1;
end

dime = size(Variable);
L = floor(dime(1)/Pck)*Pck; C = floor(dime(2)/Pck)*Pck;
ll = L*C/Pck/Pck;
A=Variable(1:L,1:C);

PackedV = zeros(L/Pck, C/Pck);
CntC = 1; CntL = 1;
for I = 1:Pck:(dime(1)-Pck+1)
    for J = 1:Pck:(dime(2)-Pck+1)
        PackedV(CntL,CntC) = sum(sum(A(I:I+Pck-1,J:J+Pck-1)));
        CntC = CntC+1;
    end
    CntL = CntL+1; CntC = 1;
end

if (SameSize == 1)  %on remet à la taille standard en assignant la valeur du "super-pixel" aux PckxPck pixels
    PackedVar = Nan(dime(1),dime(2));
    for ind1 = 1:Pck:L
        for ind2 = 1:Pck:C
            PackedVar(ind1:ind1+Pck-1,ind2:ind2+Pck-1) = ones(Pck,Pck)*PackedV((ind1-1)/Pck+1,(ind2-1)/Pck+1);
        end
    end
elseif (SameSize == 2) %fenêtre glissante
    PackedVar = NaN(dime(1),dime(2));
    for I = floor(Pck/2)+1:(dime(1)-floor(Pck/2))
        for J = floor(Pck/2)+1:(dime(2)-floor(Pck/2))
            PackedVar(I,J) = sum(sum(Variable(I+floor(-Pck/2)+1:I+floor(Pck/2),J+floor(-Pck/2)+1:J+floor(Pck/2))));
        end
    end
else 
    PackedVar(1:L/Pck,1:C/Pck) = PackedV;
end
