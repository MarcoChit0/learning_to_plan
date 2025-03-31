; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 22495

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child362 child310 - child
    bread150 bread376 - bread-portion
    content105 content72 - content-portion
    tray44 - tray
    table296 table193 table279 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray44 kitchen)
     (at_kitchen_bread bread150)
     (at_kitchen_bread bread376)
     (at_kitchen_content content105)
     (at_kitchen_content content72)
     (no_gluten_bread bread150)
     (no_gluten_content content72)
     (allergic_gluten child310)
     (not_allergic_gluten child362)
     (waiting child362 table193)
     (waiting child310 table296)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child362)
     (served child310)
    )
  )
)
