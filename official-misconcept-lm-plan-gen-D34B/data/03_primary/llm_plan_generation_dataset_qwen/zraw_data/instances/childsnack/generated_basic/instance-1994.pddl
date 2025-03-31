; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 308682

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child474 child152 - child
    bread468 bread333 - bread-portion
    content40 content70 - content-portion
    tray329 - tray
    table102 table101 table372 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray329 kitchen)
     (at_kitchen_bread bread468)
     (at_kitchen_bread bread333)
     (at_kitchen_content content40)
     (at_kitchen_content content70)
     (no_gluten_bread bread468)
     (no_gluten_content content40)
     (allergic_gluten child152)
     (not_allergic_gluten child474)
     (waiting child474 table101)
     (waiting child152 table372)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child474)
     (served child152)
    )
  )
)
