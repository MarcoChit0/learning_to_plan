; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 428388

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child65 - child
    bread304 - bread-portion
    content469 - content-portion
    tray94 tray377 - tray
    table329 table341 table478 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray94 kitchen)
     (at tray377 kitchen)
     (at_kitchen_bread bread304)
     (at_kitchen_content content469)
     (not_allergic_gluten child65)
     (waiting child65 table478)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child65)
    )
  )
)
