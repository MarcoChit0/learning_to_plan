; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 619124

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child186 child366 - child
    bread142 bread351 - bread-portion
    content433 content39 - content-portion
    tray145 tray153 - tray
    table155 table217 table195 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray145 kitchen)
     (at tray153 kitchen)
     (at_kitchen_bread bread142)
     (at_kitchen_bread bread351)
     (at_kitchen_content content433)
     (at_kitchen_content content39)
     (not_allergic_gluten child366)
     (not_allergic_gluten child186)
     (waiting child186 table195)
     (waiting child366 table155)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child186)
     (served child366)
    )
  )
)
