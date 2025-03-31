; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 316408

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child6 child338 - child
    bread401 bread94 - bread-portion
    content451 content169 - content-portion
    tray437 tray34 tray30 tray205 - tray
    table111 table43 table380 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray437 kitchen)
     (at tray34 kitchen)
     (at tray30 kitchen)
     (at tray205 kitchen)
     (at_kitchen_bread bread401)
     (at_kitchen_bread bread94)
     (at_kitchen_content content451)
     (at_kitchen_content content169)
     (no_gluten_bread bread401)
     (no_gluten_content content451)
     (allergic_gluten child338)
     (not_allergic_gluten child6)
     (waiting child6 table380)
     (waiting child338 table43)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child6)
     (served child338)
    )
  )
)
