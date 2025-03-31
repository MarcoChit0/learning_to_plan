; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 261284

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child481 child479 child444 - child
    bread362 bread382 bread204 - bread-portion
    content288 content180 content287 - content-portion
    tray346 - tray
    table3 table114 table233 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray346 kitchen)
     (at_kitchen_bread bread362)
     (at_kitchen_bread bread382)
     (at_kitchen_bread bread204)
     (at_kitchen_content content288)
     (at_kitchen_content content180)
     (at_kitchen_content content287)
     (not_allergic_gluten child481)
     (not_allergic_gluten child444)
     (not_allergic_gluten child479)
     (waiting child481 table233)
     (waiting child479 table233)
     (waiting child444 table114)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child481)
     (served child479)
     (served child444)
    )
  )
)
