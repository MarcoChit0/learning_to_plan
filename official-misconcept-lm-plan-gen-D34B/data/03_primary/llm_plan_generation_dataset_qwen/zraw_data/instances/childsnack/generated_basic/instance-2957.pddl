; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 869118

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child496 - child
    bread495 - bread-portion
    content141 - content-portion
    tray446 tray288 tray60 tray469 - tray
    table21 table168 table52 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray446 kitchen)
     (at tray288 kitchen)
     (at tray60 kitchen)
     (at tray469 kitchen)
     (at_kitchen_bread bread495)
     (at_kitchen_content content141)
     (not_allergic_gluten child496)
     (waiting child496 table21)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child496)
    )
  )
)
