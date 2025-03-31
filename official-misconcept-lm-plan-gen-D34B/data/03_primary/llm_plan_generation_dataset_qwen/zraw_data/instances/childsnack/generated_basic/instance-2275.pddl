; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 492737

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child65 child274 - child
    bread105 bread298 - bread-portion
    content39 content426 - content-portion
    tray302 - tray
    table363 table66 table308 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray302 kitchen)
     (at_kitchen_bread bread105)
     (at_kitchen_bread bread298)
     (at_kitchen_content content39)
     (at_kitchen_content content426)
     (no_gluten_bread bread298)
     (no_gluten_content content39)
     (allergic_gluten child65)
     (not_allergic_gluten child274)
     (waiting child65 table308)
     (waiting child274 table66)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child65)
     (served child274)
    )
  )
)
