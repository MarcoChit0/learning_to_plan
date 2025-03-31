; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 735624

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child457 - child
    bread19 - bread-portion
    content321 - content-portion
    tray5 - tray
    table406 table263 table286 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray5 kitchen)
     (at_kitchen_bread bread19)
     (at_kitchen_content content321)
     (not_allergic_gluten child457)
     (waiting child457 table263)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child457)
    )
  )
)
