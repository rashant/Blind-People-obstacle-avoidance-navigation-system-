hspace, angles, distances = HoughLines(ximg)

            # # Find angle
            # angle=[]
            # for _, a , distances in zip(*hough_line_peaks(hspace, angles, distances)):
            #     angle.append(a)
            
            # angles = [a*180/np.pi for a in angle]

            # # Compute difference between the two lines
            # angle_difference = np.max(angles) - np.min(angles)
            # print(angle_difference)