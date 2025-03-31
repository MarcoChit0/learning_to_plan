; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 779480

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child477 child74 child117 - child
    bread488 bread35 bread174 - bread-portion
    content299 content495 content354 - content-portion
    tray178 tray275 tray127 - tray
    table47 table51 table44 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray178 kitchen)
     (at tray275 kitchen)
     (at tray127 kitchen)
     (at_kitchen_bread bread488)
     (at_kitchen_bread bread35)
     (at_kitchen_bread bread174)
     (at_kitchen_content content299)
     (at_kitchen_content content495)
     (at_kitchen_content content354)
     (not_allergic_gluten child74)
     (not_allergic_gluten child477)
     (not_allergic_gluten child117)
     (waiting child477 table51)
     (waiting child74 table44)
     (waiting child117 table44)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child477)
     (served child74)
     (served child117)
    )
  )
)
