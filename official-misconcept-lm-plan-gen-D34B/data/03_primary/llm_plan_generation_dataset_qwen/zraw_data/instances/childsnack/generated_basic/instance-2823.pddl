; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 972971

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child147 child316 - child
    bread276 bread411 - bread-portion
    content91 content232 - content-portion
    tray99 tray400 - tray
    table221 table171 table81 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray99 kitchen)
     (at tray400 kitchen)
     (at_kitchen_bread bread276)
     (at_kitchen_bread bread411)
     (at_kitchen_content content91)
     (at_kitchen_content content232)
     (not_allergic_gluten child316)
     (not_allergic_gluten child147)
     (waiting child147 table171)
     (waiting child316 table171)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child147)
     (served child316)
    )
  )
)
