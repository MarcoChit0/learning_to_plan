; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 771202

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child367 - child
    bread21 - bread-portion
    content280 - content-portion
    tray403 tray133 tray401 tray495 - tray
    table330 table6 table269 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray403 kitchen)
     (at tray133 kitchen)
     (at tray401 kitchen)
     (at tray495 kitchen)
     (at_kitchen_bread bread21)
     (at_kitchen_content content280)
     (not_allergic_gluten child367)
     (waiting child367 table330)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child367)
    )
  )
)
