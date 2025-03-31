; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 709015

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child112 - child
    bread10 - bread-portion
    content225 - content-portion
    tray401 tray201 tray53 - tray
    table107 table261 table299 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray401 kitchen)
     (at tray201 kitchen)
     (at tray53 kitchen)
     (at_kitchen_bread bread10)
     (at_kitchen_content content225)
     (not_allergic_gluten child112)
     (waiting child112 table299)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child112)
    )
  )
)
