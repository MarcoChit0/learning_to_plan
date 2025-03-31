; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 54565

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child482 - child
    bread309 - bread-portion
    content48 - content-portion
    tray73 tray486 tray356 tray179 - tray
    table286 table222 table297 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray73 kitchen)
     (at tray486 kitchen)
     (at tray356 kitchen)
     (at tray179 kitchen)
     (at_kitchen_bread bread309)
     (at_kitchen_content content48)
     (not_allergic_gluten child482)
     (waiting child482 table222)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child482)
    )
  )
)
