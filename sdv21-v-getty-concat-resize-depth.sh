torchrun --nproc_per_node=8 train.py \
  --pretrained_model_name_or_path="/fsx_laion/alvin/pretrain/stable-diffusion-2-1" \
  --train_data_dir "/fsx_laion/alvin/Dataset/getty_human_merge" \
  --use_ema \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=8 \
  --num_train_epochs=100 \
  --checkpointing_steps=1000 \
  --validation_steps=1000 \
  --dataloader_num_workers=12 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="sdv21-v-getty-concat-resize-depth" \
  --cond_reshape "resize" \
  --cond_type "depth" \
  --cond_inject "concat" \
  --prediction_type "v_prediction" \
  --seed=0 \
  --validation_prompts "A mesmerizing oil painting in the style of Van Gogh, portraying a confident woman gracefully dancing ballet in a moonlit garden, wearing an elegant tutu and performing a series of intricate leaps and twirls, 64K resolution, a fusion of artistic mastery and captivating movement." \
  "A breathtaking photograph in black and white, capturing the moment a professional soccer player strikes the ball with precision and power, with intense focus and determination, 64K resolution, a freeze-frame of athleticism and sportsmanship." \
  "A captivating watercolor illustration of a group of friends engaged in a lively conversation at a cozy café, their animated gestures and expressions conveying laughter, connection, and camaraderie, 64K resolution, a heartwarming snapshot of friendship and shared moments." \
  "An enchanting digital artwork in the style of impressionism, depicting a couple strolling hand in hand through a sunflower field, their steps light and carefree, bathed in the golden glow of the setting sun, 64K resolution, a romantic tableau of nature's beauty and human affection." \
  "A whimsical digital artwork with a touch of fantasy, featuring a young girl gracefully floating in mid-air, surrounded by a swirl of colorful balloons, her dress billowing with the wind, 64K resolution, a magical portrayal of weightlessness and childhood dreams." \
  "A captivating oil painting capturing the raw emotion of a street performer passionately playing the violin in a bustling square, his music filling the air with soul-stirring melodies that resonate with the hearts of passersby, 64K resolution, an expression of artistic expression and connection through music." \
  "An atmospheric photograph taken at a beach at sunset, showcasing a group of friends running along the shoreline, their laughter echoing in the salty breeze, leaving footprints in the sand as they embrace the carefree joy of the moment, 64K resolution, a snapshot of youthful energy and seaside serenity." \
  "A whimsical digital artwork inspired by surrealism, featuring a person gracefully floating in a vast starry sky, surrounded by celestial objects and cosmic wonders, their movements defying gravity and transcending earthly boundaries, 64K resolution, a surreal exploration of human potential and the infinite." \
  "A dynamic black and white photograph capturing the intensity and agility of a professional dancer leaping through the air with grace and precision, frozen in a moment of artistic expression, 64K resolution, a striking blend of strength, beauty, and artistic mastery." \
  "A compelling high-resolution digital photograph capturing the genuine smile of a young child, their eyes sparkling with innocence and joy, as they playfully interact with nature in a sunlit garden, 64K resolution, a heartwarming portrayal of childhood wonder and happiness." \
  "A striking black and white portrait photograph of an elderly person, their weathered face displaying a lifetime of wisdom and resilience, the lines and wrinkles telling a story of experience and character, 64K resolution, a powerful portrayal of the beauty and grace that come with age." \
  "A mesmerizing digital photograph of a ballet dancer in mid-air, captured with impeccable timing and precision, showcasing the grace, strength, and elegance of their movements, 64K resolution, an artistic tribute to the art of dance and the human body in motion." \
  "A candid street photograph of a group of friends sharing laughter and camaraderie in a bustling cafe, their expressions and gestures reflecting genuine connection and friendship, 64K resolution, a genuine portrayal of the joy and warmth that human relationships bring." \
  "An intricately detailed portrait of a young woman with piercing blue eyes, flowing auburn hair, and delicate freckles adorning her cheeks, radiating an air of mystery and allure, 64K resolution, a captivating portrayal of individuality and beauty." \
  "A realistic digital photograph capturing the dynamic energy of a skilled basketball player in action, as he dribbles the ball with precision and leaps high into the air for a powerful slam dunk, the intensity in his eyes reflecting his passion for the game, 64K resolution, a captivating display of athleticism and the thrill of basketball." \
  "A highly detailed photograph of a muscular athlete in motion, capturing the power and definition of every muscle group, sweat glistening on their determined face, 64K resolution, an awe-inspiring depiction of human strength and athleticism." \
  "A vivid illustration of a couple holding hands, their fingers intertwined with intricate detail, showcasing the tenderness and connection between them, 64K resolution, a heartwarming portrayal of human affection and companionship." \
  "A captivating street photograph taken in a bustling city, showcasing a stylish woman confidently walking along a sunlit sidewalk, her fashionable attire and poised demeanor catching the attention of passersby, 64K resolution, a glimpse into urban elegance and individuality." \
  "An intimate portrait photograph capturing the genuine laughter of a young couple enjoying a picnic in a picturesque park, their relaxed poses and genuine affection creating a moment of pure happiness, 64K resolution, a timeless depiction of love and connection." \
  "A candid photograph taken in a vibrant market, featuring a skilled artisan passionately crafting a piece of jewelry, their focused expression and precise movements revealing the artistry behind their craft, 64K resolution, a celebration of skill and creativity." \
  "An intimate candid photograph capturing the tenderness of a parent cradling their newborn baby, their eyes filled with love and wonder, the soft lighting accentuating the delicate features of the infant, 64K resolution, a heartfelt moment of pure joy and the beauty of new life." \
  "An immersive digital photograph showcasing a talented female basketball player executing a mid-air jump shot with flawless form, the ball suspended in mid-flight as his focused gaze locks onto the hoop, 64K resolution, a remarkable blend of strength, technique, and determination on the basketball court."



# --gradient_accumulation_steps=4 \
# --gradient_checkpointing \
# --data-path /fsx_laion/laion5b/laion2B-en-aesthetic-4.5plus-512/ \
# --data-path /fsx_laion/laion5b/COYO-700M-512-min-image-size200/ \
# --data-path /fsx_laion/getty_images_webdataset \