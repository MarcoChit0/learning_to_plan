; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 993496

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child167 child108 - child
    bread344 bread167 - bread-portion
    content427 content171 - content-portion
    tray28 tray492 tray41 tray478 - tray
    table467 table366 table143 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray28 kitchen)
     (at tray492 kitchen)
     (at tray41 kitchen)
     (at tray478 kitchen)
     (at_kitchen_bread bread344)
     (at_kitchen_bread bread167)
     (at_kitchen_content content427)
     (at_kitchen_content content171)
     (no_gluten_bread bread344)
     (no_gluten_content content171)
     (allergic_gluten child108)
     (not_allergic_gluten child167)
     (waiting child167 table467)
     (waiting child108 table366)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child167)
     (served child108)
    )
  )
)
