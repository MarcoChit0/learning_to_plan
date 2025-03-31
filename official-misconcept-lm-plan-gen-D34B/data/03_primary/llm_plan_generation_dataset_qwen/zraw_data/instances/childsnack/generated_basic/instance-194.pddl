; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 108446

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child115 child340 - child
    bread310 bread125 - bread-portion
    content445 content5 - content-portion
    tray436 tray36 tray409 - tray
    table167 table83 table293 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray436 kitchen)
     (at tray36 kitchen)
     (at tray409 kitchen)
     (at_kitchen_bread bread310)
     (at_kitchen_bread bread125)
     (at_kitchen_content content445)
     (at_kitchen_content content5)
     (not_allergic_gluten child340)
     (not_allergic_gluten child115)
     (waiting child115 table167)
     (waiting child340 table293)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child115)
     (served child340)
    )
  )
)
