; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 717818

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child286 child257 - child
    bread447 bread397 - bread-portion
    content280 content492 - content-portion
    tray80 tray79 - tray
    table355 table400 table480 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray80 kitchen)
     (at tray79 kitchen)
     (at_kitchen_bread bread447)
     (at_kitchen_bread bread397)
     (at_kitchen_content content280)
     (at_kitchen_content content492)
     (no_gluten_bread bread397)
     (no_gluten_content content492)
     (allergic_gluten child257)
     (not_allergic_gluten child286)
     (waiting child286 table400)
     (waiting child257 table400)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child286)
     (served child257)
    )
  )
)
