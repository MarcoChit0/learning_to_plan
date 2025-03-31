; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 84140

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child320 child73 - child
    bread208 bread317 - bread-portion
    content400 content72 - content-portion
    tray269 tray279 - tray
    table345 table327 table340 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray269 kitchen)
     (at tray279 kitchen)
     (at_kitchen_bread bread208)
     (at_kitchen_bread bread317)
     (at_kitchen_content content400)
     (at_kitchen_content content72)
     (no_gluten_bread bread208)
     (no_gluten_content content400)
     (allergic_gluten child73)
     (not_allergic_gluten child320)
     (waiting child320 table327)
     (waiting child73 table340)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child320)
     (served child73)
    )
  )
)
