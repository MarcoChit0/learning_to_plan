; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 687923

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child82 child89 - child
    bread492 bread421 - bread-portion
    content337 content248 - content-portion
    tray149 tray25 - tray
    table373 table317 table190 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray149 kitchen)
     (at tray25 kitchen)
     (at_kitchen_bread bread492)
     (at_kitchen_bread bread421)
     (at_kitchen_content content337)
     (at_kitchen_content content248)
     (no_gluten_bread bread421)
     (no_gluten_content content248)
     (allergic_gluten child82)
     (not_allergic_gluten child89)
     (waiting child82 table317)
     (waiting child89 table373)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child82)
     (served child89)
    )
  )
)
