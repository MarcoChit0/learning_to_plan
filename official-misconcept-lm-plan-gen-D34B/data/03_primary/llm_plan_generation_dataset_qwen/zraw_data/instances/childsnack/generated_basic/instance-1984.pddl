; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 772554

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child13 child356 - child
    bread327 bread38 - bread-portion
    content169 content489 - content-portion
    tray67 - tray
    table101 table365 table391 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray67 kitchen)
     (at_kitchen_bread bread327)
     (at_kitchen_bread bread38)
     (at_kitchen_content content169)
     (at_kitchen_content content489)
     (no_gluten_bread bread38)
     (no_gluten_content content169)
     (allergic_gluten child13)
     (not_allergic_gluten child356)
     (waiting child13 table101)
     (waiting child356 table365)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child13)
     (served child356)
    )
  )
)
