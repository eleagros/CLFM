function[rot]=rota(b)

w = b*pi/180;

rot=[1 0 0 0;
      0 cos(2*w) -sin(2*w) 0;
      0 sin(2*w)  cos(2*w) 0;
      0  0          0       1];
  
  return