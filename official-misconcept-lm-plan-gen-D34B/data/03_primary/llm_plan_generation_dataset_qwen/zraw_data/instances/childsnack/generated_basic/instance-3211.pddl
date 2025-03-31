; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 626977

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child264 child270 - child
    bread213 bread27 - bread-portion
    content402 content67 - content-portion
    tray124 tray142 tray304 tray488 - tray
    table119 table157 table264 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray124 kitchen)
     (at tray142 kitchen)
     (at tray304 kitchen)
     (at tray488 kitchen)
     (at_kitchen_bread bread213)
     (at_kitchen_bread bread27)
     (at_kitchen_content content402)
     (at_kitchen_content content67)
     (no_gluten_bread bread27)
     (no_gluten_content content67)
     (allergic_gluten child270)
     (not_allergic_gluten child264)
     (waiting child264 table119)
     (waiting child270 table157)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child264)
     (served child270)
    )
  )
)
