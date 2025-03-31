; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 897727

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child69 child341 - child
    bread342 bread14 - bread-portion
    content397 content222 - content-portion
    tray322 tray38 - tray
    table384 table147 table138 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray322 kitchen)
     (at tray38 kitchen)
     (at_kitchen_bread bread342)
     (at_kitchen_bread bread14)
     (at_kitchen_content content397)
     (at_kitchen_content content222)
     (not_allergic_gluten child69)
     (not_allergic_gluten child341)
     (waiting child69 table384)
     (waiting child341 table147)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child69)
     (served child341)
    )
  )
)
