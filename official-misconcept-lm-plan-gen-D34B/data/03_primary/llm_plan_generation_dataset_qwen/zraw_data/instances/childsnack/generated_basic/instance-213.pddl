; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 939675

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child486 child138 - child
    bread261 bread153 - bread-portion
    content70 content72 - content-portion
    tray252 tray456 tray203 - tray
    table237 table35 table251 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray252 kitchen)
     (at tray456 kitchen)
     (at tray203 kitchen)
     (at_kitchen_bread bread261)
     (at_kitchen_bread bread153)
     (at_kitchen_content content70)
     (at_kitchen_content content72)
     (not_allergic_gluten child138)
     (not_allergic_gluten child486)
     (waiting child486 table237)
     (waiting child138 table251)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child486)
     (served child138)
    )
  )
)
