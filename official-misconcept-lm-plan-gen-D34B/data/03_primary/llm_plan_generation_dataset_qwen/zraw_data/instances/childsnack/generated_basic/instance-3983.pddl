; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 343712

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child33 - child
    bread427 - bread-portion
    content37 - content-portion
    tray157 tray28 tray127 - tray
    table334 table137 table438 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray157 kitchen)
     (at tray28 kitchen)
     (at tray127 kitchen)
     (at_kitchen_bread bread427)
     (at_kitchen_content content37)
     (not_allergic_gluten child33)
     (waiting child33 table438)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child33)
    )
  )
)
