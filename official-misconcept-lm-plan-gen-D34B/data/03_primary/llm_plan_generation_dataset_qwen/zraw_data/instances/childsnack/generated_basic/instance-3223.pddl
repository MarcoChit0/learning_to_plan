; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 479724

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child330 child117 - child
    bread296 bread235 - bread-portion
    content244 content366 - content-portion
    tray458 tray260 tray287 tray54 - tray
    table21 table277 table139 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray458 kitchen)
     (at tray260 kitchen)
     (at tray287 kitchen)
     (at tray54 kitchen)
     (at_kitchen_bread bread296)
     (at_kitchen_bread bread235)
     (at_kitchen_content content244)
     (at_kitchen_content content366)
     (no_gluten_bread bread296)
     (no_gluten_content content244)
     (allergic_gluten child117)
     (not_allergic_gluten child330)
     (waiting child330 table21)
     (waiting child117 table139)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child330)
     (served child117)
    )
  )
)
