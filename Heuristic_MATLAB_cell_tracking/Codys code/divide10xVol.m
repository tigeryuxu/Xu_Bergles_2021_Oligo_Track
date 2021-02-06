function [vertsABCD,vtops,vbots] = divide10xVol(verticesS)
v1t = verticesS(1,:);
v2t = verticesS(2,:);
v3t = verticesS(3,:);
v4t = verticesS(4,:);

h = calcEuclid(v1t,v2t);
b = v1t(1) - v2t(1);
a = sqrt(h^2 - b^2);
halfa = a/2;
v12 = [0 v1t(2) v1t(3)+halfa];

h = calcEuclid(v3t,v4t);
b = v3t(1) - v4t(1);
a = sqrt(h^2 - b^2);
halfa = a/2;
v34 = [0 v3t(2) v3t(3)+halfa];

h = calcEuclid(v1t,v3t);
b = v1t(2) - v3t(2);
a = sqrt(h^2 - b^2);
halfa = a/2;
v13 = [v1t(1) 0 v1t(3)+halfa];

h = calcEuclid(v2t,v4t);
b = v2t(2) - v4t(2);
a = sqrt(h^2 - b^2);
halfa = a/2;
v24 = [v2t(1) 0 v2t(3)+halfa];

zdiff = v3t(3) - v2t(3);
vc = [0 0 v3t(3)-(zdiff/2)];

vtops = [v12; v13; v34; v24; vc];
vbots = [v12(1:2),v12(3)+300; v13(1:2),v13(3)+300; v34(1:2),v34(3)+300; v24(1:2),v24(3)+300; vc(1:2),vc(3)+300];

vertsBlockA = [v1t; v12; v13; vc; v1t(1:2),v1t(3)+300; v12(1:2),v12(3)+300; v13(1:2),v13(3)+300; vc(1:2),vc(3)+300];
vertsBlockB = [v12; v2t; vc; v24; v12(1:2),v12(3)+300; v2t(1:2),v2t(3)+300; vc(1:2),vc(3)+300; v24(1:2),v24(3)+300];
vertsBlockC = [v13; vc; v3t; v34; v13(1:2),v13(3)+300; vc(1:2),vc(3)+300; v3t(1:2),v3t(3)+300; v34(1:2),v34(3)+300];
vertsBlockD = [vc; v24; v34; v4t; vc(1:2),vc(3)+300; v24(1:2),v24(3)+300; v34(1:2),v34(3)+300; v4t(1:2),v4t(3)+300];

vertsABCD = {vertsBlockA; vertsBlockB; vertsBlockC; vertsBlockD};
end