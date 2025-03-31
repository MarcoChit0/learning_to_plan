; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 561651

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child66 child452 - child
    bread100 bread254 - bread-portion
    content477 content262 - content-portion
    tray42 tray49 - tray
    table423 table82 table213 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray42 kitchen)
     (at tray49 kitchen)
     (at_kitchen_bread bread100)
     (at_kitchen_bread bread254)
     (at_kitchen_content content477)
     (at_kitchen_content content262)
     (no_gluten_bread bread254)
     (no_gluten_content content477)
     (allergic_gluten child452)
     (not_allergic_gluten child66)
     (waiting child66 table423)
     (waiting child452 table82)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child66)
     (served child452)
    )
  )
)
