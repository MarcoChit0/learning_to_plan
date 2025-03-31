; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 1

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child69 child194 - child
    bread426 bread250 - bread-portion
    content247 content26 - content-portion
    tray47 - tray
    table251 table120 table281 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray47 kitchen)
     (at_kitchen_bread bread426)
     (at_kitchen_bread bread250)
     (at_kitchen_content content247)
     (at_kitchen_content content26)
     (no_gluten_bread bread426)
     (no_gluten_content content247)
     (allergic_gluten child194)
     (not_allergic_gluten child69)
     (waiting child69 table251)
     (waiting child194 table120)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child69)
     (served child194)
    )
  )
)
