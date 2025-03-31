; child-snack task with 3 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 741799

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child362 child364 child231 - child
    bread152 bread222 bread428 - bread-portion
    content432 content206 content140 - content-portion
    tray479 tray363 tray64 - tray
    table238 table487 table489 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray479 kitchen)
     (at tray363 kitchen)
     (at tray64 kitchen)
     (at_kitchen_bread bread152)
     (at_kitchen_bread bread222)
     (at_kitchen_bread bread428)
     (at_kitchen_content content432)
     (at_kitchen_content content206)
     (at_kitchen_content content140)
     (no_gluten_bread bread428)
     (no_gluten_content content206)
     (allergic_gluten child362)
     (not_allergic_gluten child364)
     (not_allergic_gluten child231)
     (waiting child362 table238)
     (waiting child364 table238)
     (waiting child231 table238)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child362)
     (served child364)
     (served child231)
    )
  )
)
