; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 126549

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child211 - child
    bread287 - bread-portion
    content294 - content-portion
    tray232 - tray
    table21 table106 table112 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray232 kitchen)
     (at_kitchen_bread bread287)
     (at_kitchen_content content294)
     (not_allergic_gluten child211)
     (waiting child211 table112)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child211)
    )
  )
)
