; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 537224

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child379 child318 - child
    bread455 bread105 - bread-portion
    content69 content253 - content-portion
    tray166 - tray
    table419 table271 table230 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray166 kitchen)
     (at_kitchen_bread bread455)
     (at_kitchen_bread bread105)
     (at_kitchen_content content69)
     (at_kitchen_content content253)
     (no_gluten_bread bread105)
     (no_gluten_content content69)
     (allergic_gluten child379)
     (not_allergic_gluten child318)
     (waiting child379 table419)
     (waiting child318 table230)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child379)
     (served child318)
    )
  )
)
