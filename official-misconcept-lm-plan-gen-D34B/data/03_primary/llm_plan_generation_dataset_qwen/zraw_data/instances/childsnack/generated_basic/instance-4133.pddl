; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 975626

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child131 child415 - child
    bread37 bread378 - bread-portion
    content380 content117 - content-portion
    tray288 tray197 tray133 - tray
    table90 table133 table492 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray288 kitchen)
     (at tray197 kitchen)
     (at tray133 kitchen)
     (at_kitchen_bread bread37)
     (at_kitchen_bread bread378)
     (at_kitchen_content content380)
     (at_kitchen_content content117)
     (not_allergic_gluten child131)
     (not_allergic_gluten child415)
     (waiting child131 table133)
     (waiting child415 table492)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child131)
     (served child415)
    )
  )
)
