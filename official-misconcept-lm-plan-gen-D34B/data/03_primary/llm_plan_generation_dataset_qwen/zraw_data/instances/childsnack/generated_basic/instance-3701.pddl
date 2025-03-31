; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 64636

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child84 child298 - child
    bread146 bread44 - bread-portion
    content447 content222 - content-portion
    tray404 tray422 - tray
    table249 table290 table270 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray404 kitchen)
     (at tray422 kitchen)
     (at_kitchen_bread bread146)
     (at_kitchen_bread bread44)
     (at_kitchen_content content447)
     (at_kitchen_content content222)
     (no_gluten_bread bread44)
     (no_gluten_content content222)
     (allergic_gluten child298)
     (not_allergic_gluten child84)
     (waiting child84 table249)
     (waiting child298 table290)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child84)
     (served child298)
    )
  )
)
