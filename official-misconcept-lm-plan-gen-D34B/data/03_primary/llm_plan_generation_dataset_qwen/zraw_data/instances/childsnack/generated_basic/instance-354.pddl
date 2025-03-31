; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 210450

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child495 - child
    bread454 - bread-portion
    content256 - content-portion
    tray158 tray179 tray180 tray246 - tray
    table171 table203 table205 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray158 kitchen)
     (at tray179 kitchen)
     (at tray180 kitchen)
     (at tray246 kitchen)
     (at_kitchen_bread bread454)
     (at_kitchen_content content256)
     (not_allergic_gluten child495)
     (waiting child495 table203)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child495)
    )
  )
)
