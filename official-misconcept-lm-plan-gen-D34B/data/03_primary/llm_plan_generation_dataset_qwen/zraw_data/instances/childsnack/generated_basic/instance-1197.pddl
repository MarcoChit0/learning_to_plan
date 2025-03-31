; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 252651

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child17 child192 child252 - child
    bread134 bread263 bread173 - bread-portion
    content184 content280 content181 - content-portion
    tray422 tray66 tray243 - tray
    table26 table40 table30 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray422 kitchen)
     (at tray66 kitchen)
     (at tray243 kitchen)
     (at_kitchen_bread bread134)
     (at_kitchen_bread bread263)
     (at_kitchen_bread bread173)
     (at_kitchen_content content184)
     (at_kitchen_content content280)
     (at_kitchen_content content181)
     (not_allergic_gluten child252)
     (not_allergic_gluten child17)
     (not_allergic_gluten child192)
     (waiting child17 table40)
     (waiting child192 table40)
     (waiting child252 table30)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child17)
     (served child192)
     (served child252)
    )
  )
)
