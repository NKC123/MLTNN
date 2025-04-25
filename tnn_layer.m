function [output] = tnn_layer(x_data,D,A_1,b_1,A_D,b_D,A_d,b_d)

sigmaf = @(z) max(z,0);
%sigmaf = @(z)tanh(z);
output = zeros(1,size(x_data,2));

if D < 2
    print('D cannot be smaller than 2')
elseif D == 2
    layer_0 = A_1*x_data+b_1;
    output = A_D*sigmaf(layer_0) + b_D;
else
    for j = 1:size(x_data,2)
        x = x_data(:,j);
        layer_0 = A_1*x+b_1;
        layer_c = layer_0;
        for i = 2:D-1
            ii = i-1;
            layer_f = A_d(:,:,ii)*sigmaf(layer_c) + b_d(:,ii);
            layer_c = layer_f;
        end
        output(j) = A_D*sigmaf(layer_c) + b_D;
    end
end