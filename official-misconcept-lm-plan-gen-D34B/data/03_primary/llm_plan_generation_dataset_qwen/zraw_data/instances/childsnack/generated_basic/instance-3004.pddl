; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 202845

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child157 child474 - child
    bread133 bread172 - bread-portion
    content426 content4 - content-portion
    tray10 tray419 tray184 tray237 - tray
    table207 table88 table79 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray10 kitchen)
     (at tray419 kitchen)
     (at tray184 kitchen)
     (at tray237 kitchen)
     (at_kitchen_bread bread133)
     (at_kitchen_bread bread172)
     (at_kitchen_content content426)
     (at_kitchen_content content4)
     (not_allergic_gluten child157)
     (not_allergic_gluten child474)
     (waiting child157 table207)
     (waiting child474 table88)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child157)
     (served child474)
    )
  )
)
