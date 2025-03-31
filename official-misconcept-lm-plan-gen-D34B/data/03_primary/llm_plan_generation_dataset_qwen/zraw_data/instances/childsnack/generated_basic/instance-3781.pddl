; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 738519

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child475 - child
    bread405 - bread-portion
    content319 - content-portion
    tray90 - tray
    table81 table493 table335 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray90 kitchen)
     (at_kitchen_bread bread405)
     (at_kitchen_content content319)
     (not_allergic_gluten child475)
     (waiting child475 table81)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child475)
    )
  )
)
