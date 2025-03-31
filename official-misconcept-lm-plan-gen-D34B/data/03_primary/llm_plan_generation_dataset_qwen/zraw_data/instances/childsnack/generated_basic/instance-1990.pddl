; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 24471

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child8 child251 - child
    bread215 bread437 - bread-portion
    content34 content293 - content-portion
    tray324 - tray
    table32 table341 table14 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray324 kitchen)
     (at_kitchen_bread bread215)
     (at_kitchen_bread bread437)
     (at_kitchen_content content34)
     (at_kitchen_content content293)
     (no_gluten_bread bread437)
     (no_gluten_content content293)
     (allergic_gluten child8)
     (not_allergic_gluten child251)
     (waiting child8 table14)
     (waiting child251 table32)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child8)
     (served child251)
    )
  )
)
