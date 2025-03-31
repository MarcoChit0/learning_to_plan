; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 386456

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child376 - child
    bread464 - bread-portion
    content462 - content-portion
    tray433 tray432 - tray
    table369 table251 table420 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray433 kitchen)
     (at tray432 kitchen)
     (at_kitchen_bread bread464)
     (at_kitchen_content content462)
     (not_allergic_gluten child376)
     (waiting child376 table369)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child376)
    )
  )
)
