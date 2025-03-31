; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 724662

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child295 child111 - child
    bread73 bread264 - bread-portion
    content174 content3 - content-portion
    tray345 tray148 - tray
    table289 table162 table57 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray345 kitchen)
     (at tray148 kitchen)
     (at_kitchen_bread bread73)
     (at_kitchen_bread bread264)
     (at_kitchen_content content174)
     (at_kitchen_content content3)
     (not_allergic_gluten child111)
     (not_allergic_gluten child295)
     (waiting child295 table289)
     (waiting child111 table289)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child295)
     (served child111)
    )
  )
)
