; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 862901

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child380 child56 - child
    bread167 bread210 - bread-portion
    content329 content160 - content-portion
    tray415 tray6 tray439 tray244 - tray
    table347 table101 table299 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray415 kitchen)
     (at tray6 kitchen)
     (at tray439 kitchen)
     (at tray244 kitchen)
     (at_kitchen_bread bread167)
     (at_kitchen_bread bread210)
     (at_kitchen_content content329)
     (at_kitchen_content content160)
     (no_gluten_bread bread210)
     (no_gluten_content content160)
     (allergic_gluten child380)
     (not_allergic_gluten child56)
     (waiting child380 table347)
     (waiting child56 table299)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child380)
     (served child56)
    )
  )
)
