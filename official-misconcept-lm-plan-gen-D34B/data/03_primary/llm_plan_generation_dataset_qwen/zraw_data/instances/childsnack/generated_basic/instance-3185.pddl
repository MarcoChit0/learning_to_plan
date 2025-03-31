; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 777007

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child344 child159 - child
    bread377 bread164 - bread-portion
    content71 content61 - content-portion
    tray297 tray280 - tray
    table252 table214 table163 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray297 kitchen)
     (at tray280 kitchen)
     (at_kitchen_bread bread377)
     (at_kitchen_bread bread164)
     (at_kitchen_content content71)
     (at_kitchen_content content61)
     (not_allergic_gluten child344)
     (not_allergic_gluten child159)
     (waiting child344 table163)
     (waiting child159 table252)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child344)
     (served child159)
    )
  )
)
