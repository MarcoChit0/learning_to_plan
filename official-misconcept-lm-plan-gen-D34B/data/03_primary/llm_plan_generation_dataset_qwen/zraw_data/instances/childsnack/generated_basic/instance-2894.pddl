; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 656509

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child282 child93 - child
    bread492 bread23 - bread-portion
    content94 content198 - content-portion
    tray483 tray102 tray436 - tray
    table244 table115 table317 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray483 kitchen)
     (at tray102 kitchen)
     (at tray436 kitchen)
     (at_kitchen_bread bread492)
     (at_kitchen_bread bread23)
     (at_kitchen_content content94)
     (at_kitchen_content content198)
     (no_gluten_bread bread23)
     (no_gluten_content content198)
     (allergic_gluten child93)
     (not_allergic_gluten child282)
     (waiting child282 table317)
     (waiting child93 table317)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child282)
     (served child93)
    )
  )
)
