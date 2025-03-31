; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 35457

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child361 child195 - child
    bread15 bread188 - bread-portion
    content425 content56 - content-portion
    tray307 tray318 tray35 - tray
    table64 table421 table293 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray307 kitchen)
     (at tray318 kitchen)
     (at tray35 kitchen)
     (at_kitchen_bread bread15)
     (at_kitchen_bread bread188)
     (at_kitchen_content content425)
     (at_kitchen_content content56)
     (no_gluten_bread bread15)
     (no_gluten_content content425)
     (allergic_gluten child195)
     (not_allergic_gluten child361)
     (waiting child361 table293)
     (waiting child195 table421)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child361)
     (served child195)
    )
  )
)
