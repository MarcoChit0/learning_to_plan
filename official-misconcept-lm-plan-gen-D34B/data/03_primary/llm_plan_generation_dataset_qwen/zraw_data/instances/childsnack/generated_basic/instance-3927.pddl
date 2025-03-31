; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 383149

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child447 child44 - child
    bread35 bread296 - bread-portion
    content40 content347 - content-portion
    tray391 - tray
    table365 table473 table295 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray391 kitchen)
     (at_kitchen_bread bread35)
     (at_kitchen_bread bread296)
     (at_kitchen_content content40)
     (at_kitchen_content content347)
     (no_gluten_bread bread296)
     (no_gluten_content content347)
     (allergic_gluten child447)
     (not_allergic_gluten child44)
     (waiting child447 table295)
     (waiting child44 table295)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child447)
     (served child44)
    )
  )
)
