; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 623628

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child355 child24 - child
    bread391 bread425 - bread-portion
    content280 content126 - content-portion
    tray367 tray16 tray431 - tray
    table398 table434 table39 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray367 kitchen)
     (at tray16 kitchen)
     (at tray431 kitchen)
     (at_kitchen_bread bread391)
     (at_kitchen_bread bread425)
     (at_kitchen_content content280)
     (at_kitchen_content content126)
     (no_gluten_bread bread391)
     (no_gluten_content content280)
     (allergic_gluten child24)
     (not_allergic_gluten child355)
     (waiting child355 table398)
     (waiting child24 table398)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child355)
     (served child24)
    )
  )
)
