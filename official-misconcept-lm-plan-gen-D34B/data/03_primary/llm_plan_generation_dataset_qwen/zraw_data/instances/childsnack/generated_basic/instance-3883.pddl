; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 536060

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child398 - child
    bread481 - bread-portion
    content404 - content-portion
    tray302 tray80 - tray
    table353 table218 table231 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray302 kitchen)
     (at tray80 kitchen)
     (at_kitchen_bread bread481)
     (at_kitchen_content content404)
     (not_allergic_gluten child398)
     (waiting child398 table231)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child398)
    )
  )
)
