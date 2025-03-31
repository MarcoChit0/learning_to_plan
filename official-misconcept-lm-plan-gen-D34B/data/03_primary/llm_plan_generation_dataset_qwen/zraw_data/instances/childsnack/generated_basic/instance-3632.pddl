; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 646584

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child163 child403 - child
    bread426 bread409 - bread-portion
    content491 content236 - content-portion
    tray148 tray244 tray256 tray83 - tray
    table221 table136 table101 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray148 kitchen)
     (at tray244 kitchen)
     (at tray256 kitchen)
     (at tray83 kitchen)
     (at_kitchen_bread bread426)
     (at_kitchen_bread bread409)
     (at_kitchen_content content491)
     (at_kitchen_content content236)
     (no_gluten_bread bread409)
     (no_gluten_content content236)
     (allergic_gluten child403)
     (not_allergic_gluten child163)
     (waiting child163 table221)
     (waiting child403 table136)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child163)
     (served child403)
    )
  )
)
