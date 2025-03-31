; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 340381

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child496 child250 - child
    bread382 bread65 - bread-portion
    content371 content438 - content-portion
    tray433 - tray
    table271 table120 table354 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray433 kitchen)
     (at_kitchen_bread bread382)
     (at_kitchen_bread bread65)
     (at_kitchen_content content371)
     (at_kitchen_content content438)
     (no_gluten_bread bread65)
     (no_gluten_content content371)
     (allergic_gluten child250)
     (not_allergic_gluten child496)
     (waiting child496 table271)
     (waiting child250 table120)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child496)
     (served child250)
    )
  )
)
