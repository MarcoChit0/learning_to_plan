; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 243771

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child90 child316 - child
    bread176 bread362 - bread-portion
    content308 content106 - content-portion
    tray493 tray251 tray484 - tray
    table425 table256 table398 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray493 kitchen)
     (at tray251 kitchen)
     (at tray484 kitchen)
     (at_kitchen_bread bread176)
     (at_kitchen_bread bread362)
     (at_kitchen_content content308)
     (at_kitchen_content content106)
     (no_gluten_bread bread362)
     (no_gluten_content content308)
     (allergic_gluten child316)
     (not_allergic_gluten child90)
     (waiting child90 table425)
     (waiting child316 table425)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child90)
     (served child316)
    )
  )
)
