; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 164156

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child123 - child
    bread432 - bread-portion
    content24 - content-portion
    tray50 tray462 tray482 - tray
    table393 table110 table28 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray50 kitchen)
     (at tray462 kitchen)
     (at tray482 kitchen)
     (at_kitchen_bread bread432)
     (at_kitchen_content content24)
     (not_allergic_gluten child123)
     (waiting child123 table110)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child123)
    )
  )
)
