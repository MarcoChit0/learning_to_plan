; child-snack task with 2 children and 0.8 gluten factor 
; constant factor of 1.7
; random seed: 588720

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child117 child38 - child
    bread425 bread350 - bread-portion
    content192 content487 - content-portion
    tray370 tray69 - tray
    table418 table179 table380 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray370 kitchen)
     (at tray69 kitchen)
     (at_kitchen_bread bread425)
     (at_kitchen_bread bread350)
     (at_kitchen_content content192)
     (at_kitchen_content content487)
     (no_gluten_bread bread425)
     (no_gluten_content content487)
     (allergic_gluten child38)
     (not_allergic_gluten child117)
     (waiting child117 table380)
     (waiting child38 table380)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child117)
     (served child38)
    )
  )
)
