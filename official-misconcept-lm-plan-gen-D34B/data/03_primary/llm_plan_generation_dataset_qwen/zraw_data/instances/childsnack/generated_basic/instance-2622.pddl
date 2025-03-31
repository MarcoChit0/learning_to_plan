; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 175510

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child297 - child
    bread415 - bread-portion
    content57 - content-portion
    tray218 tray372 tray260 - tray
    table96 table149 table101 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray218 kitchen)
     (at tray372 kitchen)
     (at tray260 kitchen)
     (at_kitchen_bread bread415)
     (at_kitchen_content content57)
     (not_allergic_gluten child297)
     (waiting child297 table96)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child297)
    )
  )
)
