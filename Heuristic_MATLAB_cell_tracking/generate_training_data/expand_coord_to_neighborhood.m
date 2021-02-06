function [expanded, lin_inds] =  expand_coord_to_neighborhood(coord, lower, upper, im_size)
    expanded = [];
    
    lin_inds = [];
    
    for x = -lower:upper
        for y = -lower:upper
            for z = -lower: upper
                new_idx = [coord(1) + x, coord(2) + y, coord(3) + z];
                
                %% ensure doesn't go out of bounds
                if new_idx(1) <= 0; new_idx(1) = 1;  end
                if new_idx(1) > im_size(1); new_idx(1) = im_size(1); end
                
                if new_idx(2) <= 0; new_idx(2) = 1;  end
                if new_idx(2) > im_size(2); new_idx(2) = im_size(2); end
                
                if new_idx(3) <= 0; new_idx(3) = 1;  end
                if new_idx(3) > im_size(3); new_idx(3) = im_size(3); end
                
                linear = sub2ind(im_size, new_idx(1), new_idx(2), new_idx(3));
                
                
                expanded = [expanded; new_idx];
                lin_inds = [lin_inds; linear];
            end
        end
    end
    
    
    