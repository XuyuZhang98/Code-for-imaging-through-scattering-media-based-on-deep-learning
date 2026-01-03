function FOV = fov(Y, X, D)
    % 参数检查
    if D > min(X, Y)
        error('The diameter D cannot be larger than the dimensions X or Y.');
    end
    
    % 生成网格坐标
    [XX, YY] = meshgrid(1:X, 1:Y);

    % 计算中心点
    center_X0 = floor(X / 2) + 1;
    center_Y0 = floor(Y / 2) + 1;

    % 生成圆形激光光束的FOV矩阵
    FOV = ((XX - center_X0).^2 + (YY - center_Y0).^2) < (D / 2)^2;

    % 转换为双精度
    FOV = double(FOV);
end