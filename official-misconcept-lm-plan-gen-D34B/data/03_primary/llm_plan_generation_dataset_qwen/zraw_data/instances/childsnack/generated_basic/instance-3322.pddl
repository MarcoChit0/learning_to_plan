; child-snack task with 2 children and 0.5 gluten factor 
; constant factor of 1.7
; random seed: 612243

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child474 child371 - child
    bread171 bread391 - bread-portion
    content240 content444 - content-portion
    tray328 tray113 tray494 - tray
    table321 table385 table245 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray328 kitchen)
     (at tray113 kitchen)
     (at tray494 kitchen)
     (at_kitchen_bread bread171)
     (at_kitchen_bread bread391)
     (at_kitchen_content content240)
     (at_kitchen_content content444)
     (no_gluten_bread bread171)
     (no_gluten_content content240)
     (allergic_gluten child474)
     (not_allergic_gluten child371)
     (waiting child474 table321)
     (waiting child371 table245)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child474)
     (served child371)
    )
  )
)
