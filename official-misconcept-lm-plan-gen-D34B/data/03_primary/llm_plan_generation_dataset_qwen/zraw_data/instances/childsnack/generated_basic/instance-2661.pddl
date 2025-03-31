; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 904215

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child75 - child
    bread331 - bread-portion
    content268 - content-portion
    tray178 tray124 tray414 tray273 - tray
    table130 table183 table457 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray178 kitchen)
     (at tray124 kitchen)
     (at tray414 kitchen)
     (at tray273 kitchen)
     (at_kitchen_bread bread331)
     (at_kitchen_content content268)
     (not_allergic_gluten child75)
     (waiting child75 table457)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child75)
    )
  )
)
