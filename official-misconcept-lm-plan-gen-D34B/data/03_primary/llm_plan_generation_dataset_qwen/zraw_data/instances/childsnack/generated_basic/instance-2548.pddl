; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 368376

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child491 - child
    bread249 - bread-portion
    content352 - content-portion
    tray93 tray433 tray134 - tray
    table254 table259 table483 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray93 kitchen)
     (at tray433 kitchen)
     (at tray134 kitchen)
     (at_kitchen_bread bread249)
     (at_kitchen_content content352)
     (not_allergic_gluten child491)
     (waiting child491 table254)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child491)
    )
  )
)
