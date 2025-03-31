; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 643298

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child252 child486 - child
    bread46 bread295 - bread-portion
    content172 content454 - content-portion
    tray18 - tray
    table415 table394 table224 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray18 kitchen)
     (at_kitchen_bread bread46)
     (at_kitchen_bread bread295)
     (at_kitchen_content content172)
     (at_kitchen_content content454)
     (no_gluten_bread bread46)
     (no_gluten_content content454)
     (allergic_gluten child252)
     (not_allergic_gluten child486)
     (waiting child252 table415)
     (waiting child486 table415)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child252)
     (served child486)
    )
  )
)
