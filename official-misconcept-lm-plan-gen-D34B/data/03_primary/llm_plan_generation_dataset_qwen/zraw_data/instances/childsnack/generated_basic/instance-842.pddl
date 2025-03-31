; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 842836

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child417 child202 - child
    bread225 bread385 - bread-portion
    content264 content174 - content-portion
    tray445 tray273 - tray
    table193 table31 table286 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray445 kitchen)
     (at tray273 kitchen)
     (at_kitchen_bread bread225)
     (at_kitchen_bread bread385)
     (at_kitchen_content content264)
     (at_kitchen_content content174)
     (no_gluten_bread bread385)
     (no_gluten_content content264)
     (allergic_gluten child202)
     (not_allergic_gluten child417)
     (waiting child417 table286)
     (waiting child202 table193)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child417)
     (served child202)
    )
  )
)
