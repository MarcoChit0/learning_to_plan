; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 186

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child439 child497 - child
    bread86 bread246 - bread-portion
    content46 content218 - content-portion
    tray383 - tray
    table125 table484 table57 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray383 kitchen)
     (at_kitchen_bread bread86)
     (at_kitchen_bread bread246)
     (at_kitchen_content content46)
     (at_kitchen_content content218)
     (not_allergic_gluten child439)
     (not_allergic_gluten child497)
     (waiting child439 table484)
     (waiting child497 table57)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child439)
     (served child497)
    )
  )
)
