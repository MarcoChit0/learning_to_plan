; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 499715

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child209 - child
    bread306 - bread-portion
    content95 - content-portion
    tray357 tray66 - tray
    table174 table119 table199 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray357 kitchen)
     (at tray66 kitchen)
     (at_kitchen_bread bread306)
     (at_kitchen_content content95)
     (not_allergic_gluten child209)
     (waiting child209 table199)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child209)
    )
  )
)
