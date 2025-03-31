; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 974423

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child83 - child
    bread498 - bread-portion
    content75 - content-portion
    tray443 - tray
    table455 table374 table77 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray443 kitchen)
     (at_kitchen_bread bread498)
     (at_kitchen_content content75)
     (not_allergic_gluten child83)
     (waiting child83 table77)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child83)
    )
  )
)
