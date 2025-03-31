; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 930083

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child136 child241 child478 - child
    bread388 bread448 bread429 - bread-portion
    content273 content34 content63 - content-portion
    tray79 - tray
    table348 table232 table448 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray79 kitchen)
     (at_kitchen_bread bread388)
     (at_kitchen_bread bread448)
     (at_kitchen_bread bread429)
     (at_kitchen_content content273)
     (at_kitchen_content content34)
     (at_kitchen_content content63)
     (not_allergic_gluten child136)
     (not_allergic_gluten child241)
     (not_allergic_gluten child478)
     (waiting child136 table232)
     (waiting child241 table448)
     (waiting child478 table448)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child136)
     (served child241)
     (served child478)
    )
  )
)
