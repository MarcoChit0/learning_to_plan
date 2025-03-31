; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 324862

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child308 child402 - child
    bread117 bread489 - bread-portion
    content399 content342 - content-portion
    tray344 tray461 - tray
    table149 table49 table472 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray344 kitchen)
     (at tray461 kitchen)
     (at_kitchen_bread bread117)
     (at_kitchen_bread bread489)
     (at_kitchen_content content399)
     (at_kitchen_content content342)
     (not_allergic_gluten child402)
     (not_allergic_gluten child308)
     (waiting child308 table149)
     (waiting child402 table149)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child308)
     (served child402)
    )
  )
)
