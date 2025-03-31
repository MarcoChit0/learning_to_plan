; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 327740

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child121 child391 child377 - child
    bread127 bread182 bread395 - bread-portion
    content91 content184 content70 - content-portion
    tray330 tray420 tray344 - tray
    table218 table428 table369 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray330 kitchen)
     (at tray420 kitchen)
     (at tray344 kitchen)
     (at_kitchen_bread bread127)
     (at_kitchen_bread bread182)
     (at_kitchen_bread bread395)
     (at_kitchen_content content91)
     (at_kitchen_content content184)
     (at_kitchen_content content70)
     (not_allergic_gluten child121)
     (not_allergic_gluten child391)
     (not_allergic_gluten child377)
     (waiting child121 table218)
     (waiting child391 table218)
     (waiting child377 table428)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child121)
     (served child391)
     (served child377)
    )
  )
)
