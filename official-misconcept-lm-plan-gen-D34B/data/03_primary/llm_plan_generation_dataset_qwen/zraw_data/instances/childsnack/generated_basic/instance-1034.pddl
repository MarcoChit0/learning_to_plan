; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 917

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child308 child409 - child
    bread139 bread421 - bread-portion
    content267 content434 - content-portion
    tray368 tray320 tray93 - tray
    table34 table155 table181 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray368 kitchen)
     (at tray320 kitchen)
     (at tray93 kitchen)
     (at_kitchen_bread bread139)
     (at_kitchen_bread bread421)
     (at_kitchen_content content267)
     (at_kitchen_content content434)
     (not_allergic_gluten child308)
     (not_allergic_gluten child409)
     (waiting child308 table181)
     (waiting child409 table181)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child308)
     (served child409)
    )
  )
)
