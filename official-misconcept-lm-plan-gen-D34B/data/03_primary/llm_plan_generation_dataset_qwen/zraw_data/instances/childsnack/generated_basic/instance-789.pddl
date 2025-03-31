; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 212454

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child8 - child
    bread348 - bread-portion
    content177 - content-portion
    tray368 tray231 - tray
    table83 table176 table24 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray368 kitchen)
     (at tray231 kitchen)
     (at_kitchen_bread bread348)
     (at_kitchen_content content177)
     (not_allergic_gluten child8)
     (waiting child8 table24)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child8)
    )
  )
)
