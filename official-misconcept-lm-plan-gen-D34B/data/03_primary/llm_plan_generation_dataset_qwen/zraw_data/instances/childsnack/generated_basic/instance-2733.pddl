; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 909971

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child328 child127 - child
    bread80 bread410 - bread-portion
    content377 content499 - content-portion
    tray201 - tray
    table225 table150 table495 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray201 kitchen)
     (at_kitchen_bread bread80)
     (at_kitchen_bread bread410)
     (at_kitchen_content content377)
     (at_kitchen_content content499)
     (no_gluten_bread bread410)
     (no_gluten_content content377)
     (allergic_gluten child328)
     (not_allergic_gluten child127)
     (waiting child328 table495)
     (waiting child127 table495)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child328)
     (served child127)
    )
  )
)
