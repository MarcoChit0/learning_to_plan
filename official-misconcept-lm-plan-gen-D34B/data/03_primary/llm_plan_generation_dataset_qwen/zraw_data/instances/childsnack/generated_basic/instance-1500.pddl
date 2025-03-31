; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 787084

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child237 child437 - child
    bread489 bread485 - bread-portion
    content156 content152 - content-portion
    tray396 tray392 tray51 - tray
    table51 table242 table253 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray396 kitchen)
     (at tray392 kitchen)
     (at tray51 kitchen)
     (at_kitchen_bread bread489)
     (at_kitchen_bread bread485)
     (at_kitchen_content content156)
     (at_kitchen_content content152)
     (no_gluten_bread bread485)
     (no_gluten_content content152)
     (allergic_gluten child237)
     (not_allergic_gluten child437)
     (waiting child237 table242)
     (waiting child437 table242)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child237)
     (served child437)
    )
  )
)
