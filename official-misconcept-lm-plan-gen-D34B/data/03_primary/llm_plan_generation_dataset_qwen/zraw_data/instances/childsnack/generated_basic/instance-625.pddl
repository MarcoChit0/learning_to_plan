; child-snack task with 3 children and 0.7 gluten factor 
; constant factor of 1.7
; random seed: 414145

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child70 child468 child282 - child
    bread382 bread14 bread391 - bread-portion
    content50 content469 content133 - content-portion
    tray328 tray286 - tray
    table453 table300 table358 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray328 kitchen)
     (at tray286 kitchen)
     (at_kitchen_bread bread382)
     (at_kitchen_bread bread14)
     (at_kitchen_bread bread391)
     (at_kitchen_content content50)
     (at_kitchen_content content469)
     (at_kitchen_content content133)
     (no_gluten_bread bread382)
     (no_gluten_bread bread14)
     (no_gluten_content content469)
     (no_gluten_content content133)
     (allergic_gluten child70)
     (allergic_gluten child282)
     (not_allergic_gluten child468)
     (waiting child70 table358)
     (waiting child468 table358)
     (waiting child282 table300)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child70)
     (served child468)
     (served child282)
    )
  )
)
