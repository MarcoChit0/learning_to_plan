; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 132016

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child191 - child
    bread260 - bread-portion
    content153 - content-portion
    tray218 tray450 tray373 tray317 - tray
    table193 table347 table279 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray218 kitchen)
     (at tray450 kitchen)
     (at tray373 kitchen)
     (at tray317 kitchen)
     (at_kitchen_bread bread260)
     (at_kitchen_content content153)
     (not_allergic_gluten child191)
     (waiting child191 table193)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child191)
    )
  )
)
