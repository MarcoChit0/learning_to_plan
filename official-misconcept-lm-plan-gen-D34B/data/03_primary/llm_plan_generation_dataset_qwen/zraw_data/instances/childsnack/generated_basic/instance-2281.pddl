; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 510399

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child331 child335 - child
    bread199 bread71 - bread-portion
    content231 content125 - content-portion
    tray142 - tray
    table80 table347 table488 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray142 kitchen)
     (at_kitchen_bread bread199)
     (at_kitchen_bread bread71)
     (at_kitchen_content content231)
     (at_kitchen_content content125)
     (no_gluten_bread bread199)
     (no_gluten_content content125)
     (allergic_gluten child331)
     (not_allergic_gluten child335)
     (waiting child331 table488)
     (waiting child335 table80)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child331)
     (served child335)
    )
  )
)
