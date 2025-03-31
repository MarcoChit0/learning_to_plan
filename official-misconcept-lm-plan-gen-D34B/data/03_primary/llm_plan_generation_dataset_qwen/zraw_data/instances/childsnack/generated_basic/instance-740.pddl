; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 40421

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child269 child98 - child
    bread404 bread474 - bread-portion
    content4 content405 - content-portion
    tray5 tray308 - tray
    table156 table348 table0 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray5 kitchen)
     (at tray308 kitchen)
     (at_kitchen_bread bread404)
     (at_kitchen_bread bread474)
     (at_kitchen_content content4)
     (at_kitchen_content content405)
     (no_gluten_bread bread404)
     (no_gluten_content content405)
     (allergic_gluten child98)
     (not_allergic_gluten child269)
     (waiting child269 table156)
     (waiting child98 table0)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child269)
     (served child98)
    )
  )
)
