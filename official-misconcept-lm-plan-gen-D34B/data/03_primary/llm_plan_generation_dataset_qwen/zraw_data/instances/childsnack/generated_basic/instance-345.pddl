; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 150648

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child172 - child
    bread250 - bread-portion
    content216 - content-portion
    tray203 tray351 tray33 tray218 - tray
    table157 table69 table198 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray203 kitchen)
     (at tray351 kitchen)
     (at tray33 kitchen)
     (at tray218 kitchen)
     (at_kitchen_bread bread250)
     (at_kitchen_content content216)
     (not_allergic_gluten child172)
     (waiting child172 table157)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child172)
    )
  )
)
