; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 73005

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child377 child428 - child
    bread242 bread59 - bread-portion
    content293 content440 - content-portion
    tray155 - tray
    table29 table72 table222 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray155 kitchen)
     (at_kitchen_bread bread242)
     (at_kitchen_bread bread59)
     (at_kitchen_content content293)
     (at_kitchen_content content440)
     (no_gluten_bread bread242)
     (no_gluten_content content440)
     (allergic_gluten child428)
     (not_allergic_gluten child377)
     (waiting child377 table72)
     (waiting child428 table72)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child377)
     (served child428)
    )
  )
)
