; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 948723

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child463 child176 - child
    bread194 bread149 - bread-portion
    content476 content487 - content-portion
    tray56 tray333 tray172 tray316 - tray
    table403 table455 table77 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray56 kitchen)
     (at tray333 kitchen)
     (at tray172 kitchen)
     (at tray316 kitchen)
     (at_kitchen_bread bread194)
     (at_kitchen_bread bread149)
     (at_kitchen_content content476)
     (at_kitchen_content content487)
     (not_allergic_gluten child463)
     (not_allergic_gluten child176)
     (waiting child463 table77)
     (waiting child176 table77)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child463)
     (served child176)
    )
  )
)
