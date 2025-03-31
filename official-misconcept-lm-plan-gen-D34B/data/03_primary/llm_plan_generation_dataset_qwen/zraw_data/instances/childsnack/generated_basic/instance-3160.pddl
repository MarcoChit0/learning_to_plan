; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 806183

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child185 child241 - child
    bread401 bread105 - bread-portion
    content181 content155 - content-portion
    tray218 tray461 - tray
    table348 table265 table227 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray218 kitchen)
     (at tray461 kitchen)
     (at_kitchen_bread bread401)
     (at_kitchen_bread bread105)
     (at_kitchen_content content181)
     (at_kitchen_content content155)
     (not_allergic_gluten child241)
     (not_allergic_gluten child185)
     (waiting child185 table265)
     (waiting child241 table348)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child185)
     (served child241)
    )
  )
)
