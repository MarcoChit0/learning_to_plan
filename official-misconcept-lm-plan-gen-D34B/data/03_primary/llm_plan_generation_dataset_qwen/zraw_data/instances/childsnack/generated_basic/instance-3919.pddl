; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 291509

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child338 child34 - child
    bread370 bread218 - bread-portion
    content369 content163 - content-portion
    tray276 - tray
    table282 table352 table228 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray276 kitchen)
     (at_kitchen_bread bread370)
     (at_kitchen_bread bread218)
     (at_kitchen_content content369)
     (at_kitchen_content content163)
     (no_gluten_bread bread218)
     (no_gluten_content content163)
     (allergic_gluten child338)
     (not_allergic_gluten child34)
     (waiting child338 table352)
     (waiting child34 table228)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child338)
     (served child34)
    )
  )
)
