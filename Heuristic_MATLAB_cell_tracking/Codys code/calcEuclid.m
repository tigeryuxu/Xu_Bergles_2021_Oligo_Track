function d = calcEuclid(coords1,coords2,scale)
    if nargin<3
        scale = [1 1 1];
    end
    if size(coords1,2) < 3
        coords1 = [coords1,1];
    end
    if size(coords2,2) < 3
        coords2 = [coords2,1];
    end
    coords1 = coords1 .* scale;
    coords2 = coords2 .* scale;
    d = sqrt((coords1(1) - coords2(1))^2 ...
        + (coords1(2) - coords2(2))^2 ...
        + (coords1(3) - coords2(3))^2);
end