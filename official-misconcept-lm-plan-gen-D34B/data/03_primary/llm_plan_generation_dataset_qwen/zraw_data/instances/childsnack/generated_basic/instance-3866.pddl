; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 113489

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child309 - child
    bread420 - bread-portion
    content166 - content-portion
    tray227 tray425 - tray
    table89 table337 table409 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray227 kitchen)
     (at tray425 kitchen)
     (at_kitchen_bread bread420)
     (at_kitchen_content content166)
     (not_allergic_gluten child309)
     (waiting child309 table409)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child309)
    )
  )
)
