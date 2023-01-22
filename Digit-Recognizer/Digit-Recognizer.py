
    while (cap.isOpened()):
        ret, img = cap.read()
        img, contours, thresh = get_img_contour_thresh(img)
        ans1 = ''
        ans2 = ''
        ans3 = ''
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                # print(predict(w_from_model,b_from_model,contour))
                x, y, w, h = cv2.boundingRect(contour)
                # newImage = thresh[y - 15:y + h + 15, x - 15:x + w +15]
                newImage = thresh[y:y + h, x:x + w]
                newImage = cv2.resize(newImage, (28, 28))
                newImage = np.array(newImage)
                newImage = newImage.flatten()
                newImage = newImage.reshape(newImage.shape[0], 1)
                ans1 = Digit_Recognizer_LR.predict(w_LR, b_LR, newImage)
                ans2 = Digit_Recognizer_NN.predict_nn(d2, newImage)
                ans3 = Digit_Recognizer_DL.predict(d3, newImage)

        x, y, w, h = 0, 0, 300, 300
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Logistic Regression : " + str(ans1), (10, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "Shallow Network :  " + str(ans2), (10, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "Deep Network :  " + str(ans3), (10, 380),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Frame", img)
        cv2.imshow("Contours", thresh)
        k = cv2.waitKey(10)
        if k == 27:
            break
