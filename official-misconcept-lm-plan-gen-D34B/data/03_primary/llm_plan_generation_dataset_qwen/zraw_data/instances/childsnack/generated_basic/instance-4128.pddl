; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 85641

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child111 child215 - child
    bread426 bread182 - bread-portion
    content332 content96 - content-portion
    tray411 tray38 tray491 - tray
    table333 table439 table489 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray411 kitchen)
     (at tray38 kitchen)
     (at tray491 kitchen)
     (at_kitchen_bread bread426)
     (at_kitchen_bread bread182)
     (at_kitchen_content content332)
     (at_kitchen_content content96)
     (not_allergic_gluten child215)
     (not_allergic_gluten child111)
     (waiting child111 table489)
     (waiting child215 table439)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child111)
     (served child215)
    )
  )
)
