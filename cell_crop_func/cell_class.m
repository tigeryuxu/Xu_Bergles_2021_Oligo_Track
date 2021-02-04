classdef cell_class
    properties
        voxelIdxList
        centroid
        cell_number
        confidence_color
    end
    methods
        %% constructor functions
        function obj = cell_class(voxelIdxList, centroid, cell_number, confidence_color)
            if nargin == 3
                obj.voxelIdxList = voxelIdxList;
                obj.centroid = centroid;
                obj.cell_number = cell_number;
                
            elseif nargin == 4
                obj.voxelIdxList = voxelIdxList;
                obj.centroid = centroid;
                obj.cell_number = cell_number;
                obj.confidence_color = confidence_color;
            end
            
            
            
            
        end
    end
end