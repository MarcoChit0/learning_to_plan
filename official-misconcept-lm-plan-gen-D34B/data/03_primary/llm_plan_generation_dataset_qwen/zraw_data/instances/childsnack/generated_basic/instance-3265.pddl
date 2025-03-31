; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 32658

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child344 child230 - child
    bread132 bread292 - bread-portion
    content280 content474 - content-portion
    tray289 tray15 tray298 tray174 - tray
    table466 table363 table370 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray289 kitchen)
     (at tray15 kitchen)
     (at tray298 kitchen)
     (at tray174 kitchen)
     (at_kitchen_bread bread132)
     (at_kitchen_bread bread292)
     (at_kitchen_content content280)
     (at_kitchen_content content474)
     (no_gluten_bread bread292)
     (no_gluten_content content474)
     (allergic_gluten child344)
     (not_allergic_gluten child230)
     (waiting child344 table370)
     (waiting child230 table466)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child344)
     (served child230)
    )
  )
)
