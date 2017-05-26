function accu = do_accuracy_evaluate( depth_inpaint, ground_truth, thr)

[H, W] = size(ground_truth);
if(H~=size(depth_inpaint, 1) || W~=size(depth_inpaint,2))
    depth_inpaint = imresize(depth_inpaint, [H W]);
end

img = max(ground_truth ./ depth_inpaint,  depth_inpaint ./ ground_truth);
accu_id1 = find(img < thr);
accu_id2 = find(img < thr^2);
accu_id3 = find(img < thr^3);

accu = [length(accu_id1), length(accu_id2), length(accu_id3)] / H / W;

end

