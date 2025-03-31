; child-snack task with 3 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 824182

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child299 child89 child315 - child
    bread57 bread347 bread87 - bread-portion
    content488 content114 content405 - content-portion
    tray427 tray465 tray187 tray219 - tray
    table14 table243 table269 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray427 kitchen)
     (at tray465 kitchen)
     (at tray187 kitchen)
     (at tray219 kitchen)
     (at_kitchen_bread bread57)
     (at_kitchen_bread bread347)
     (at_kitchen_bread bread87)
     (at_kitchen_content content488)
     (at_kitchen_content content114)
     (at_kitchen_content content405)
     (no_gluten_bread bread347)
     (no_gluten_content content114)
     (allergic_gluten child89)
     (not_allergic_gluten child315)
     (not_allergic_gluten child299)
     (waiting child299 table243)
     (waiting child89 table14)
     (waiting child315 table14)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child299)
     (served child89)
     (served child315)
    )
  )
)
