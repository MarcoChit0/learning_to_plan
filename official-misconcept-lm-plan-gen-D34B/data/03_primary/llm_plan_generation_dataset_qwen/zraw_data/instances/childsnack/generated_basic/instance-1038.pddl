; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 783151

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child348 child211 - child
    bread388 bread170 - bread-portion
    content87 content70 - content-portion
    tray276 tray443 tray107 - tray
    table103 table250 table164 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray276 kitchen)
     (at tray443 kitchen)
     (at tray107 kitchen)
     (at_kitchen_bread bread388)
     (at_kitchen_bread bread170)
     (at_kitchen_content content87)
     (at_kitchen_content content70)
     (not_allergic_gluten child348)
     (not_allergic_gluten child211)
     (waiting child348 table164)
     (waiting child211 table250)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child348)
     (served child211)
    )
  )
)
