%% Face Detection and Tracking Using the KLT Algorithm
% This example shows how to automatically detect and track a face using
% feature points. The approach in this example keeps track of the face even
% when the person tilts his or her head, or moves toward or away from the
% camera.
%
%   Copyright 2014 The MathWorks, Inc.

%% Introduction
% Object detection and tracking are important in many computer vision
% applications including activity recognition, automotive safety, and
% surveillance. In this example, you will develop a simple face tracking
% system by dividing the tracking problem into three parts:   
% 
% # Detect a face
% # Identify facial features to track 
% # Track the face

%% Detect a Face
% First, you must detect the face. Use the |vision.CascadeObjectDetector|
% System object(TM) to detect the location of a face in a video frame. The
% cascade object detector uses the Viola-Jones detection algorithm and a
% trained classification model for detection. By default, the detector is
% configured to detect faces, but it can be used to detect other types of
% objects. 

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();      %can detect mouths too
% to be increased to make sure we only get 1 face (defaults to 4)
faceDetector.MergeThreshold = 10;

% Read a video frame and run the face detector.
fileName = 'tmpObamaAddress.avi';
videoFileReader = vision.VideoFileReader(fileName, 'AudioOutputPort',true);  
videoFileWriter = vision.VideoFileWriter (strcat('bbox_',fileName),'AudioInputPort',true,'FrameRate',videoFileReader.info.VideoFrameRate);
%videoFileWriter.VideoCompressor='MJPEG Compressor';
videoFileWriter.VideoCompressor='DV Video Encoder';


frame = 0;
% move past empty frames - go to ~5 seconds into video
while (frame < 160)
    [videoFrame, audioFrame]      = step(videoFileReader);
    %videoFrame      = step(videoFileReader);
    frame = frame + 1;
end;

bbox            = step(faceDetector, videoFrame);
% Draw the returned bounding box around the detected face.
videoFrame = insertShape(videoFrame, 'Rectangle', bbox);
%figure; imshow(videoFrame); title('Detected face');


% Convert the first box into a list of 4 points
% This is needed to be able to visualize the rotation of the object.
bboxPoints = bbox2points(bbox(1, :));

%%
% To track the face over time, this example uses the Kanade-Lucas-Tomasi
% (KLT) algorithm. While it is possible to use the cascade object detector
% on every frame, it is computationally expensive. It may also fail to
% detect the face, when the subject turns or tilts his head. This
% limitation comes from the type of trained classification model used for
% detection. The example detects the face only once, and then the KLT
% algorithm tracks the face across the video frames. 

%% Identify Facial Features To Track
% The KLT algorithm tracks a set of feature points across the video frames.
% Once the detection locates the face, the next step in the example
% identifies feature points that can be reliably tracked.  This example
% uses the standard, "good features to track" proposed by Shi and Tomasi. 

% Detect feature points in the face region.
%find feature points in the face
%points = detectMSERFeatures(rgb2gray(videoFrame), 'ROI', bbox);
%points = detectFASTFeatures(rgb2gray(videoFrame), 'ROI', bbox);

% points1 = detectBRISKFeatures(videoFrame(:,:,1), 'ROI', bbox);
% points2 = detectBRISKFeatures(videoFrame(:,:,2), 'ROI', bbox);
% points3 = detectBRISKFeatures(videoFrame(:,:,3), 'ROI', bbox);
% points1 = detectSURFFeatures(videoFrame(:,:,1), 'ROI', bbox);
% points2 = detectSURFFeatures(videoFrame(:,:,2), 'ROI', bbox);
% points3 = detectSURFFeatures(videoFrame(:,:,3), 'ROI', bbox);
%find most here in blue channel in eigen features, which is darker than red or green
points = detectMinEigenFeatures(videoFrame(:,:,3), 'ROI', bbox);

% points1 = detectMinEigenFeatures(videoFrame(:,:,1), 'ROI', bbox);
% points2 = detectMinEigenFeatures(videoFrame(:,:,2), 'ROI', bbox);
% points3 = detectMinEigenFeatures(videoFrame(:,:,3), 'ROI', bbox);
% Display the detected points.
% figure, imshow(videoFrame(:,:,1)), hold on;
% plot(points1);
% figure, imshow(videoFrame(:,:,2)), hold on;
% plot(points2);
% figure, imshow(videoFrame(:,:,3)), hold on;
% plot(points3);


%% Initialize a Tracker to Track the Points
% With the feature points identified, you can now use the
% |vision.PointTracker| System object to track them. For each point in the
% previous frame, the point tracker attempts to find the corresponding
% point in the current frame. Then the |estimateGeometricTransform|
% function is used to estimate the translation, rotation, and scale between
% the old points and the new points. This transformation is applied to the
% bounding box around the face.

% Create a point tracker and enable the bidirectional error constraint to
% make it more robust in the presence of noise and clutter.
pointTracker = vision.PointTracker('MaxBidirectionalError', 12);

% Initialize the tracker with the initial point locations and the initial
% video frame.
points = points.Location;
initialize(pointTracker, points, videoFrame);

%% Initialize a Video Player to Display the Results
% Create a video player object for displaying video frames.
videoPlayer  = vision.VideoPlayer('Position',...
    [100 100 [size(videoFrame, 2)+100, size(videoFrame, 1)]+30]);

%% Track the Face
% Track the points from frame to frame, and use
% |estimateGeometricTransform| function to estimate the motion of the face.

% Make a copy of the points to be used for computing the geometric
% transformation between the points in the previous and the current frames
oldPoints = points;
numOldPoints = size(points,1);
while ~isDone(videoFileReader)
    % get the next frame
    [videoFrame, audioFrame]      = step(videoFileReader);
    
    % Track the points. Note that some points may be lost.
    [points, isFound] = step(pointTracker, videoFrame);
    
    visiblePoints = points(isFound, :);
    oldInliers = oldPoints(isFound, :);
    numNewPoints = size(visiblePoints,1);
    if numNewPoints >= 2 % need at least 2 points
        
        % Estimate the geometric transformation between the old points
        % and the new points and eliminate outliers
        [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
            oldInliers, visiblePoints, 'similarity', 'MaxDistance', 12);
        
        % Apply the transformation to the bounding box points
        bboxPoints = transformPointsForward(xform, bboxPoints);
        % Insert a bounding box around the object being tracked
        bboxPolygon = reshape(bboxPoints', 1, []);
        videoFrame = insertShape(videoFrame, 'Polygon', bboxPolygon, ...
            'LineWidth', 2);
                
        % Display tracked points
        videoFramePts = insertMarker(videoFrame, visiblePoints, '+', ...
            'Color', 'white');
        %find more points if we lost too many of the old ones
        if(numNewPoints < .75*numOldPoints)
            %update bboxs - find points in each quadrant
            newPoints = reacquirePoints(videoFrame(:,:,3),bboxPoints);
            oldPoints = unique(vertcat(visiblePoints, newPoints), 'rows');
        else
            oldPoints = visiblePoints;
        end
        setPoints(pointTracker, oldPoints);        
               
    end
    %save frame without points
    step(videoFileWriter, videoFrame, audioFrame);
    % Display the annotated video frame using the video player object
    step(videoPlayer, videoFramePts);
end

% Clean up
release(videoFileReader);
release(videoFileWriter);
release(videoPlayer);
release(pointTracker);

%% Summary
% In this example, you created a simple face tracking system that
% automatically detects and tracks a single face. Try changing the input
% video, and see if you are still able to detect and track a face. Make
% sure the person is facing the camera in the initial frame for the
% detection step.

%% References
%
% Viola, Paul A. and Jones, Michael J. "Rapid Object Detection using a
% Boosted Cascade of Simple Features", IEEE CVPR, 2001.
%
% Bruce D. Lucas and Takeo Kanade. An Iterative Image Registration 
% Technique with an Application to Stereo Vision. 
% International Joint Conference on Artificial Intelligence, 1981.
%
% Carlo Tomasi and Takeo Kanade. Detection and Tracking of Point Features. 
% Carnegie Mellon University Technical Report CMU-CS-91-132, 1991.
%
% Jianbo Shi and Carlo Tomasi. Good Features to Track. 
% IEEE Conference on Computer Vision and Pattern Recognition, 1994.
%
% Zdenek Kalal, Krystian Mikolajczyk and Jiri Matas. Forward-Backward
% Error: Automatic Detection of Tracking Failures.
% International Conference on Pattern Recognition, 2010


