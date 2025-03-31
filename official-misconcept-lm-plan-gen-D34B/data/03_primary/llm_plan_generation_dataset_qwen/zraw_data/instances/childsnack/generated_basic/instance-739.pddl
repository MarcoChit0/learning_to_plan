; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 740972

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child201 child36 - child
    bread174 bread379 - bread-portion
    content339 content217 - content-portion
    tray43 tray202 - tray
    table126 table445 table170 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray43 kitchen)
     (at tray202 kitchen)
     (at_kitchen_bread bread174)
     (at_kitchen_bread bread379)
     (at_kitchen_content content339)
     (at_kitchen_content content217)
     (no_gluten_bread bread379)
     (no_gluten_content content339)
     (allergic_gluten child201)
     (not_allergic_gluten child36)
     (waiting child201 table126)
     (waiting child36 table170)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child201)
     (served child36)
    )
  )
)
