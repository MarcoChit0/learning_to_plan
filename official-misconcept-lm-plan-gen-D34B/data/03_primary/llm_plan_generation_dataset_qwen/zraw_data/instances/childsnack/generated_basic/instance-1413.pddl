; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 713571

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child494 child353 - child
    bread262 bread240 - bread-portion
    content136 content81 - content-portion
    tray310 - tray
    table271 table101 table405 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray310 kitchen)
     (at_kitchen_bread bread262)
     (at_kitchen_bread bread240)
     (at_kitchen_content content136)
     (at_kitchen_content content81)
     (no_gluten_bread bread240)
     (no_gluten_content content136)
     (allergic_gluten child353)
     (not_allergic_gluten child494)
     (waiting child494 table101)
     (waiting child353 table101)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child494)
     (served child353)
    )
  )
)
