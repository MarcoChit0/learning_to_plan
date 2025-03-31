; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 87427

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child92 - child
    bread329 - bread-portion
    content51 - content-portion
    tray479 tray235 - tray
    table242 table101 table77 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray479 kitchen)
     (at tray235 kitchen)
     (at_kitchen_bread bread329)
     (at_kitchen_content content51)
     (not_allergic_gluten child92)
     (waiting child92 table242)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child92)
    )
  )
)
