; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 378735

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child150 - child
    bread287 - bread-portion
    content391 - content-portion
    tray189 tray237 tray175 - tray
    table373 table349 table62 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray189 kitchen)
     (at tray237 kitchen)
     (at tray175 kitchen)
     (at_kitchen_bread bread287)
     (at_kitchen_content content391)
     (not_allergic_gluten child150)
     (waiting child150 table373)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child150)
    )
  )
)
