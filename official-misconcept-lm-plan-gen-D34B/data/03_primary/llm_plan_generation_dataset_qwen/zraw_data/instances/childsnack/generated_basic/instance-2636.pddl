; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 487902

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child339 - child
    bread479 - bread-portion
    content17 - content-portion
    tray226 tray245 tray105 tray41 - tray
    table398 table110 table183 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray226 kitchen)
     (at tray245 kitchen)
     (at tray105 kitchen)
     (at tray41 kitchen)
     (at_kitchen_bread bread479)
     (at_kitchen_content content17)
     (not_allergic_gluten child339)
     (waiting child339 table398)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child339)
    )
  )
)
