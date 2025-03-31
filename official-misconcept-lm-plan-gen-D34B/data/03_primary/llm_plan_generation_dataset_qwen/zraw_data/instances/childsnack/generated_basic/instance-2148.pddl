; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 985451

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child10 child142 - child
    bread72 bread479 - bread-portion
    content427 content70 - content-portion
    tray105 tray95 tray210 - tray
    table419 table72 table297 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray105 kitchen)
     (at tray95 kitchen)
     (at tray210 kitchen)
     (at_kitchen_bread bread72)
     (at_kitchen_bread bread479)
     (at_kitchen_content content427)
     (at_kitchen_content content70)
     (no_gluten_bread bread479)
     (no_gluten_content content427)
     (allergic_gluten child10)
     (not_allergic_gluten child142)
     (waiting child10 table297)
     (waiting child142 table72)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child10)
     (served child142)
    )
  )
)
