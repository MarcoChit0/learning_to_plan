; child-snack task with 2 children and 0.56 gluten factor 
; constant factor of 1.3
; random seed: 887828

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child361 child439 - child
    bread433 bread100 - bread-portion
    content32 content72 - content-portion
    tray461 tray306 - tray
    table3 table399 table244 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray461 kitchen)
     (at tray306 kitchen)
     (at_kitchen_bread bread433)
     (at_kitchen_bread bread100)
     (at_kitchen_content content32)
     (at_kitchen_content content72)
     (no_gluten_bread bread100)
     (no_gluten_content content32)
     (allergic_gluten child439)
     (not_allergic_gluten child361)
     (waiting child361 table3)
     (waiting child439 table244)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child361)
     (served child439)
    )
  )
)
