; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 536195

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child39 child150 - child
    bread54 bread248 - bread-portion
    content315 content365 - content-portion
    tray67 tray414 tray393 - tray
    table460 table35 table393 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray67 kitchen)
     (at tray414 kitchen)
     (at tray393 kitchen)
     (at_kitchen_bread bread54)
     (at_kitchen_bread bread248)
     (at_kitchen_content content315)
     (at_kitchen_content content365)
     (no_gluten_bread bread248)
     (no_gluten_content content365)
     (allergic_gluten child150)
     (not_allergic_gluten child39)
     (waiting child39 table35)
     (waiting child150 table393)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child39)
     (served child150)
    )
  )
)
