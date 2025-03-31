; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 986803

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child119 - child
    bread389 - bread-portion
    content376 - content-portion
    tray92 tray21 - tray
    table92 table413 table199 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray92 kitchen)
     (at tray21 kitchen)
     (at_kitchen_bread bread389)
     (at_kitchen_content content376)
     (not_allergic_gluten child119)
     (waiting child119 table199)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child119)
    )
  )
)
