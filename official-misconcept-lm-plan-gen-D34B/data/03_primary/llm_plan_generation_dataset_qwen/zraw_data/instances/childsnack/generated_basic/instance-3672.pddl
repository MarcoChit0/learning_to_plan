; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 801795

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child95 child263 - child
    bread450 bread56 - bread-portion
    content164 content444 - content-portion
    tray358 tray266 - tray
    table464 table141 table321 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray358 kitchen)
     (at tray266 kitchen)
     (at_kitchen_bread bread450)
     (at_kitchen_bread bread56)
     (at_kitchen_content content164)
     (at_kitchen_content content444)
     (no_gluten_bread bread450)
     (no_gluten_content content444)
     (allergic_gluten child263)
     (not_allergic_gluten child95)
     (waiting child95 table321)
     (waiting child263 table464)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child95)
     (served child263)
    )
  )
)
