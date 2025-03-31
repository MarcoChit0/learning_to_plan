; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 232324

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child184 child84 - child
    bread70 bread396 - bread-portion
    content309 content297 - content-portion
    tray313 - tray
    table327 table44 table473 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray313 kitchen)
     (at_kitchen_bread bread70)
     (at_kitchen_bread bread396)
     (at_kitchen_content content309)
     (at_kitchen_content content297)
     (no_gluten_bread bread70)
     (no_gluten_content content297)
     (allergic_gluten child184)
     (not_allergic_gluten child84)
     (waiting child184 table327)
     (waiting child84 table473)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child184)
     (served child84)
    )
  )
)
