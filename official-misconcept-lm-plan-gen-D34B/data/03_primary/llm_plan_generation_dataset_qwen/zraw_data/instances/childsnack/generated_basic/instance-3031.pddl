; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 269961

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child203 child258 - child
    bread89 bread468 - bread-portion
    content161 content12 - content-portion
    tray298 tray122 tray19 - tray
    table361 table59 table482 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray298 kitchen)
     (at tray122 kitchen)
     (at tray19 kitchen)
     (at_kitchen_bread bread89)
     (at_kitchen_bread bread468)
     (at_kitchen_content content161)
     (at_kitchen_content content12)
     (no_gluten_bread bread89)
     (no_gluten_content content12)
     (allergic_gluten child203)
     (not_allergic_gluten child258)
     (waiting child203 table361)
     (waiting child258 table482)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child203)
     (served child258)
    )
  )
)
