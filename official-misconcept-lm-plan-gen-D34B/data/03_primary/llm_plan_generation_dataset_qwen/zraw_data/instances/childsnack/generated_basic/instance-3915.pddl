; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 580598

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child131 child27 - child
    bread409 bread132 - bread-portion
    content217 content446 - content-portion
    tray481 - tray
    table19 table374 table373 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray481 kitchen)
     (at_kitchen_bread bread409)
     (at_kitchen_bread bread132)
     (at_kitchen_content content217)
     (at_kitchen_content content446)
     (no_gluten_bread bread409)
     (no_gluten_content content217)
     (allergic_gluten child27)
     (not_allergic_gluten child131)
     (waiting child131 table373)
     (waiting child27 table373)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child131)
     (served child27)
    )
  )
)
