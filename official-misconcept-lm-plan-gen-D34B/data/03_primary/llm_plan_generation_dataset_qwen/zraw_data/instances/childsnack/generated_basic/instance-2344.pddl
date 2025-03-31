; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 788

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child232 - child
    bread314 - bread-portion
    content107 - content-portion
    tray256 - tray
    table44 table225 table85 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray256 kitchen)
     (at_kitchen_bread bread314)
     (at_kitchen_content content107)
     (not_allergic_gluten child232)
     (waiting child232 table85)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child232)
    )
  )
)
