; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 638079

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child98 child269 - child
    bread379 bread81 - bread-portion
    content489 content254 - content-portion
    tray344 tray355 - tray
    table423 table241 table7 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray344 kitchen)
     (at tray355 kitchen)
     (at_kitchen_bread bread379)
     (at_kitchen_bread bread81)
     (at_kitchen_content content489)
     (at_kitchen_content content254)
     (not_allergic_gluten child98)
     (not_allergic_gluten child269)
     (waiting child98 table241)
     (waiting child269 table423)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child98)
     (served child269)
    )
  )
)
