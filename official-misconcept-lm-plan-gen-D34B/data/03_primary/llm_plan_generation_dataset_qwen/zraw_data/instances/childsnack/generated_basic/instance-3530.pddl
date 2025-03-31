; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 617409

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child341 child11 - child
    bread222 bread138 - bread-portion
    content305 content494 - content-portion
    tray453 tray272 - tray
    table495 table404 table163 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray453 kitchen)
     (at tray272 kitchen)
     (at_kitchen_bread bread222)
     (at_kitchen_bread bread138)
     (at_kitchen_content content305)
     (at_kitchen_content content494)
     (not_allergic_gluten child341)
     (not_allergic_gluten child11)
     (waiting child341 table495)
     (waiting child11 table495)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child341)
     (served child11)
    )
  )
)
