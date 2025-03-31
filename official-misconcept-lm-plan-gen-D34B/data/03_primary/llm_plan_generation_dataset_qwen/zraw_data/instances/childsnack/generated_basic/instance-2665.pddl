; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 531158

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child142 - child
    bread146 - bread-portion
    content144 - content-portion
    tray453 tray347 tray483 tray41 - tray
    table402 table182 table328 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray453 kitchen)
     (at tray347 kitchen)
     (at tray483 kitchen)
     (at tray41 kitchen)
     (at_kitchen_bread bread146)
     (at_kitchen_content content144)
     (not_allergic_gluten child142)
     (waiting child142 table402)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child142)
    )
  )
)
