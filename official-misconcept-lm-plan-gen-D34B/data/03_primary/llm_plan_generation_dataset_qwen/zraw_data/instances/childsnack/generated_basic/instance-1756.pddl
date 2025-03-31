; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 848476

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child439 - child
    bread115 - bread-portion
    content369 - content-portion
    tray344 - tray
    table197 table209 table261 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray344 kitchen)
     (at_kitchen_bread bread115)
     (at_kitchen_content content369)
     (not_allergic_gluten child439)
     (waiting child439 table261)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child439)
    )
  )
)
