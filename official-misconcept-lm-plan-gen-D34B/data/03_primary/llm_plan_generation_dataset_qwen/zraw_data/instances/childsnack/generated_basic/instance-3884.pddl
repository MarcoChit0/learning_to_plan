; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 88231

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child457 - child
    bread432 - bread-portion
    content305 - content-portion
    tray276 tray48 - tray
    table426 table163 table330 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray276 kitchen)
     (at tray48 kitchen)
     (at_kitchen_bread bread432)
     (at_kitchen_content content305)
     (not_allergic_gluten child457)
     (waiting child457 table426)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child457)
    )
  )
)
