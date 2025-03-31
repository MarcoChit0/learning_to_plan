; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 583735

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child79 child93 child50 - child
    bread5 bread309 bread11 - bread-portion
    content38 content9 content294 - content-portion
    tray80 - tray
    table432 table182 table211 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray80 kitchen)
     (at_kitchen_bread bread5)
     (at_kitchen_bread bread309)
     (at_kitchen_bread bread11)
     (at_kitchen_content content38)
     (at_kitchen_content content9)
     (at_kitchen_content content294)
     (not_allergic_gluten child50)
     (not_allergic_gluten child79)
     (not_allergic_gluten child93)
     (waiting child79 table432)
     (waiting child93 table211)
     (waiting child50 table432)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child79)
     (served child93)
     (served child50)
    )
  )
)
