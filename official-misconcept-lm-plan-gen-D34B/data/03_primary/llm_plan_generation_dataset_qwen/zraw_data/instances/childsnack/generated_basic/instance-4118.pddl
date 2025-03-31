; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 806482

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child225 child370 - child
    bread83 bread331 - bread-portion
    content489 content178 - content-portion
    tray67 tray345 tray12 - tray
    table389 table105 table398 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray67 kitchen)
     (at tray345 kitchen)
     (at tray12 kitchen)
     (at_kitchen_bread bread83)
     (at_kitchen_bread bread331)
     (at_kitchen_content content489)
     (at_kitchen_content content178)
     (not_allergic_gluten child370)
     (not_allergic_gluten child225)
     (waiting child225 table398)
     (waiting child370 table389)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child225)
     (served child370)
    )
  )
)
