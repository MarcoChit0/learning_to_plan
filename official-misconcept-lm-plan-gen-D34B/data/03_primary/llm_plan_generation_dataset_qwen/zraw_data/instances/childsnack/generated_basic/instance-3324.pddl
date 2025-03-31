; child-snack task with 2 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 819094

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child37 child7 - child
    bread390 bread437 - bread-portion
    content449 content264 - content-portion
    tray73 - tray
    table130 table98 table296 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray73 kitchen)
     (at_kitchen_bread bread390)
     (at_kitchen_bread bread437)
     (at_kitchen_content content449)
     (at_kitchen_content content264)
     (no_gluten_bread bread437)
     (no_gluten_content content264)
     (allergic_gluten child7)
     (not_allergic_gluten child37)
     (waiting child37 table130)
     (waiting child7 table98)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child37)
     (served child7)
    )
  )
)
