function error = do_error_evaluate( depth_inpaint, ground_truth )

[H, W] = size(ground_truth);
if(H~=size(depth_inpaint, 1) || W~=size(depth_inpaint,2))
    depth_inpaint = imresize(depth_inpaint, [H W]);
end

rel = sum(sum(abs(depth_inpaint-ground_truth)./ground_truth)) / H / W;

rms = sqrt(sum(sum( (depth_inpaint - ground_truth).^2 )) / H / W);

err_log = sum(sum( abs(log10(ground_truth ./ depth_inpaint)) )) / H / W;

error = [rel, rms, err_log];


end

