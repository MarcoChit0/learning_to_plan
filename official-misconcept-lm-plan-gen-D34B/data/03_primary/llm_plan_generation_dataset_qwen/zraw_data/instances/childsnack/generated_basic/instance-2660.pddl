; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 883646

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child306 - child
    bread331 - bread-portion
    content164 - content-portion
    tray484 tray218 tray48 tray53 - tray
    table98 table9 table151 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray484 kitchen)
     (at tray218 kitchen)
     (at tray48 kitchen)
     (at tray53 kitchen)
     (at_kitchen_bread bread331)
     (at_kitchen_content content164)
     (not_allergic_gluten child306)
     (waiting child306 table98)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child306)
    )
  )
)
