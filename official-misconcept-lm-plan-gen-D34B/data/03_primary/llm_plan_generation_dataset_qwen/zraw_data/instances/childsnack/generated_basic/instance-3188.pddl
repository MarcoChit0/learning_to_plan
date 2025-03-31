; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 77099

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child66 child50 - child
    bread220 bread33 - bread-portion
    content353 content116 - content-portion
    tray326 tray19 - tray
    table384 table103 table210 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray326 kitchen)
     (at tray19 kitchen)
     (at_kitchen_bread bread220)
     (at_kitchen_bread bread33)
     (at_kitchen_content content353)
     (at_kitchen_content content116)
     (no_gluten_bread bread33)
     (no_gluten_content content116)
     (allergic_gluten child66)
     (not_allergic_gluten child50)
     (waiting child66 table210)
     (waiting child50 table384)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child66)
     (served child50)
    )
  )
)
