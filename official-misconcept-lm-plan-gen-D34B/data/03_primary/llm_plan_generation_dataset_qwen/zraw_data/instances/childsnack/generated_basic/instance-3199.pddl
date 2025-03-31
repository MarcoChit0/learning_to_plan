; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 72805

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child184 child6 - child
    bread4 bread147 - bread-portion
    content329 content158 - content-portion
    tray422 tray346 - tray
    table61 table162 table494 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray422 kitchen)
     (at tray346 kitchen)
     (at_kitchen_bread bread4)
     (at_kitchen_bread bread147)
     (at_kitchen_content content329)
     (at_kitchen_content content158)
     (no_gluten_bread bread4)
     (no_gluten_content content158)
     (allergic_gluten child184)
     (not_allergic_gluten child6)
     (waiting child184 table61)
     (waiting child6 table494)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child184)
     (served child6)
    )
  )
)
