; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 508378

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child250 - child
    bread127 - bread-portion
    content351 - content-portion
    tray2 - tray
    table419 table197 table462 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray2 kitchen)
     (at_kitchen_bread bread127)
     (at_kitchen_content content351)
     (not_allergic_gluten child250)
     (waiting child250 table419)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child250)
    )
  )
)
