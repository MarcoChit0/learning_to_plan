; child-snack task with 3 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 199808

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child351 child103 child332 - child
    bread394 bread425 bread108 - bread-portion
    content70 content325 content50 - content-portion
    tray242 tray132 tray346 - tray
    table339 table6 table198 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray242 kitchen)
     (at tray132 kitchen)
     (at tray346 kitchen)
     (at_kitchen_bread bread394)
     (at_kitchen_bread bread425)
     (at_kitchen_bread bread108)
     (at_kitchen_content content70)
     (at_kitchen_content content325)
     (at_kitchen_content content50)
     (no_gluten_bread bread425)
     (no_gluten_content content70)
     (allergic_gluten child351)
     (not_allergic_gluten child103)
     (not_allergic_gluten child332)
     (waiting child351 table6)
     (waiting child103 table198)
     (waiting child332 table339)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child351)
     (served child103)
     (served child332)
    )
  )
)
