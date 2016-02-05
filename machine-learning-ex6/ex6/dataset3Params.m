function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
prediction_error = 1000;

for i = 1:length(C_vec)
    for j = 1:length(sigma_vec)
        %train the model for C_vec(i), sigma_vec(j)
        model = svmTrain(X, y, C_vec(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vec(j)));
        
        %get the predictions for the trained model
        predictions = svmPredict(model, Xval);
        
        %calculate the predictions errors
        pred_error = mean(double(predictions ~= yval));
        
        %find the minimum prediction error and set C and sigma
        if(pred_error < prediction_error)
            prediction_error = pred_error;
            C = C_vec(i);
            sigma = sigma_vec(j);
        end
    end
end
% =========================================================================

end
