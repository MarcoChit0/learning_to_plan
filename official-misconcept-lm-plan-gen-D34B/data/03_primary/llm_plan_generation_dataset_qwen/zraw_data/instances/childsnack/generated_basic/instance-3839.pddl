; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 396280

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child188 - child
    bread229 - bread-portion
    content186 - content-portion
    tray207 tray392 tray54 tray21 - tray
    table479 table220 table327 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray207 kitchen)
     (at tray392 kitchen)
     (at tray54 kitchen)
     (at tray21 kitchen)
     (at_kitchen_bread bread229)
     (at_kitchen_content content186)
     (not_allergic_gluten child188)
     (waiting child188 table479)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child188)
    )
  )
)
