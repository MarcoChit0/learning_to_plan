; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 228623

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child66 child463 - child
    bread485 bread94 - bread-portion
    content177 content122 - content-portion
    tray68 - tray
    table302 table171 table104 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray68 kitchen)
     (at_kitchen_bread bread485)
     (at_kitchen_bread bread94)
     (at_kitchen_content content177)
     (at_kitchen_content content122)
     (no_gluten_bread bread94)
     (no_gluten_content content122)
     (allergic_gluten child66)
     (not_allergic_gluten child463)
     (waiting child66 table171)
     (waiting child463 table302)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child66)
     (served child463)
    )
  )
)
