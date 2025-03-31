; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 939848

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child361 child285 - child
    bread174 bread130 - bread-portion
    content181 content405 - content-portion
    tray33 tray436 - tray
    table299 table417 table87 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray33 kitchen)
     (at tray436 kitchen)
     (at_kitchen_bread bread174)
     (at_kitchen_bread bread130)
     (at_kitchen_content content181)
     (at_kitchen_content content405)
     (not_allergic_gluten child285)
     (not_allergic_gluten child361)
     (waiting child361 table417)
     (waiting child285 table299)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child361)
     (served child285)
    )
  )
)
