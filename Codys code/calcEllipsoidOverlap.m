function section_vol = calcEllipsoidOverlap(centroid1,Rxy1,Rz1,centroid2,Rxy2,Rz2)

X=[centroid1(1),centroid2(1)];   %two ellipsoid x coordinates
Y=[centroid1(2),centroid2(2)];   %two ellipsoid y coordinates
Z=[centroid1(3),centroid2(3)];   %two ellipsoid z coordinates
ROTATIONANGLE=[0,0];
AXIS_X=[Rxy1,Rxy2];  %two ellipsoid radii 
AXIS_Y=[Rxy1,Rxy2];  
AXIS_Z=[Rz1,Rz2]; 

ranges = zeros(3,2); %prealloc

do_plot = 0; %either plot ellipsoids or not

step_number = 300000;

for i = 1:2  %display ellipsoid
    if (do_plot == 1)
        [x, y, z] = ellipsoid(X(i),Y(i),Z(i),AXIS_X(i),AXIS_Y(i),AXIS_Z(i),20);
        S = surf(x, y, z);
        alpha(.1);
        rotate(S,[0,0,1], ROTATIONANGLE(i), [X(i),Y(i),Z(i)]);
    end
    %calculate the ranges for the simulation box
    ranges(1, 1) = min(ranges(1, 1), X(i) -   max(AXIS_X(i), AXIS_Y(i))  );
    ranges(1, 2) = max(ranges(1, 2), X(i) +   max(AXIS_X(i), AXIS_Y(i))  );

    ranges(2, 1) = min(ranges(2, 1), Y(i) -   max(AXIS_X(i), AXIS_Y(i))  );
    ranges(2, 2) = max(ranges(2, 2), Y(i) +   max(AXIS_X(i), AXIS_Y(i))  );

    ranges(3, 1) = min(ranges(3, 1), Z(i) - AXIS_Z(i));
    ranges(3, 2) = max(ranges(3, 2), Z(i) + AXIS_Z(i));
    if (do_plot == 1)
        hold on;
    end
end

counter = 0; %how many points targeted the intersection

for i = 1:step_number
    R = rand(3, 1).*(ranges(:, 2) - ranges(:, 1)) + ranges(:, 1); %a random point
    n = 1;
    val = calc_ellipsoid( R(1), R(2), R(3), X(n),Y(n),Z(n),AXIS_X(n),AXIS_Y(n),AXIS_Z(n),ROTATIONANGLE(n)*pi/180);
    if (val <= 1.0)
        n = 2;
        val = calc_ellipsoid( R(1), R(2), R(3), X(n),Y(n),Z(n),AXIS_X(n),AXIS_Y(n),AXIS_Z(n),ROTATIONANGLE(n)*pi/180);
        if (val <= 1.0)
            if (do_plot == 1)
                plot3(R(1), R(2), R(3), 'or', 'MarkerSize', 1, 'MarkerFaceColor','r');
            end
            counter = counter + 1;
        end
    end
end
cube_vol = 1; %the volume of the simulation box
for i=1:3
    cube_vol = cube_vol * (ranges(i, 2) - ranges(i, 1));
end

%approximated volume of the intersection
section_vol = cube_vol * counter / step_number;

% display(['Cube volume: ', num2str(cube_vol)]);
% display(['Targeted points: ', num2str(counter), ' from ', num2str(step_number)]);
% display(['Section volume: ', num2str(section_vol)]);

if (do_plot == 1)
    hold off;
end

%the function calculates a value for some given point, which shows if the
%point is inside of the ellipsoid or not.
%for a point to be inside, the value has to be <=1
function [ val ] = calc_ellipsoid( x, y, z, x0, y0, z0, a, b, c, theta)
    %x, y, z - coordinates of the point to be checked
    %x0, y0, z0 - center of the ellipsoid
    %a, b, c - axes of the ellipsoid
    %theta - angle of the rotation about the Z-axis
    x_cmp = ((x - x0)*cos(theta) + (y - y0)*sin(theta))^2   /(a^2);
    y_cmp = ((x - x0)*sin(theta) - (y - y0)*cos(theta))^2   /(b^2);
    z_cmp = (z - z0)^2 / (c^2);
    val = x_cmp + y_cmp + z_cmp;
end
end