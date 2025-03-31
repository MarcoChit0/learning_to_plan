; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 101053

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child284 child90 - child
    bread374 bread50 - bread-portion
    content166 content109 - content-portion
    tray10 - tray
    table19 table304 table330 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray10 kitchen)
     (at_kitchen_bread bread374)
     (at_kitchen_bread bread50)
     (at_kitchen_content content166)
     (at_kitchen_content content109)
     (not_allergic_gluten child90)
     (not_allergic_gluten child284)
     (waiting child284 table19)
     (waiting child90 table304)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child284)
     (served child90)
    )
  )
)
