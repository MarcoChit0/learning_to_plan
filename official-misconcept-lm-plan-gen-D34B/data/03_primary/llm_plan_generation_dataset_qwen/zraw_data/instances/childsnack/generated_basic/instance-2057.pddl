; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 943357

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child291 - child
    bread469 - bread-portion
    content157 - content-portion
    tray344 tray116 tray133 tray42 - tray
    table220 table471 table459 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray344 kitchen)
     (at tray116 kitchen)
     (at tray133 kitchen)
     (at tray42 kitchen)
     (at_kitchen_bread bread469)
     (at_kitchen_content content157)
     (not_allergic_gluten child291)
     (waiting child291 table220)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child291)
    )
  )
)
