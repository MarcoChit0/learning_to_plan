; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 560303

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child432 child252 - child
    bread232 bread108 - bread-portion
    content261 content142 - content-portion
    tray48 tray324 tray151 tray486 - tray
    table410 table232 table489 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray48 kitchen)
     (at tray324 kitchen)
     (at tray151 kitchen)
     (at tray486 kitchen)
     (at_kitchen_bread bread232)
     (at_kitchen_bread bread108)
     (at_kitchen_content content261)
     (at_kitchen_content content142)
     (not_allergic_gluten child252)
     (not_allergic_gluten child432)
     (waiting child432 table489)
     (waiting child252 table410)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child432)
     (served child252)
    )
  )
)
