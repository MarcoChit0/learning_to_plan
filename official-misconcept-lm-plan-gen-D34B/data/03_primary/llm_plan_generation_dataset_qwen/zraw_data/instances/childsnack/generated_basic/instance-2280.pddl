; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 240703

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child147 child461 - child
    bread167 bread255 - bread-portion
    content120 content359 - content-portion
    tray35 - tray
    table227 table296 table157 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray35 kitchen)
     (at_kitchen_bread bread167)
     (at_kitchen_bread bread255)
     (at_kitchen_content content120)
     (at_kitchen_content content359)
     (no_gluten_bread bread167)
     (no_gluten_content content359)
     (allergic_gluten child147)
     (not_allergic_gluten child461)
     (waiting child147 table157)
     (waiting child461 table296)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child147)
     (served child461)
    )
  )
)
