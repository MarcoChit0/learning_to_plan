; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 208207

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child140 - child
    bread426 - bread-portion
    content409 - content-portion
    tray473 tray407 tray6 tray362 - tray
    table5 table79 table163 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray473 kitchen)
     (at tray407 kitchen)
     (at tray6 kitchen)
     (at tray362 kitchen)
     (at_kitchen_bread bread426)
     (at_kitchen_content content409)
     (not_allergic_gluten child140)
     (waiting child140 table5)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child140)
    )
  )
)
