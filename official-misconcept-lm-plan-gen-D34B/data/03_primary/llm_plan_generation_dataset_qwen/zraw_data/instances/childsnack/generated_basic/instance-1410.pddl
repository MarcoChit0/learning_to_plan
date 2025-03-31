; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 657798

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child138 child81 - child
    bread8 bread4 - bread-portion
    content265 content270 - content-portion
    tray412 - tray
    table233 table373 table194 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray412 kitchen)
     (at_kitchen_bread bread8)
     (at_kitchen_bread bread4)
     (at_kitchen_content content265)
     (at_kitchen_content content270)
     (no_gluten_bread bread4)
     (no_gluten_content content270)
     (allergic_gluten child138)
     (not_allergic_gluten child81)
     (waiting child138 table373)
     (waiting child81 table233)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child138)
     (served child81)
    )
  )
)
