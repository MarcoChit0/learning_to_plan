; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 920502

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child400 child86 - child
    bread243 bread350 - bread-portion
    content465 content364 - content-portion
    tray141 tray219 tray150 - tray
    table284 table161 table173 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray141 kitchen)
     (at tray219 kitchen)
     (at tray150 kitchen)
     (at_kitchen_bread bread243)
     (at_kitchen_bread bread350)
     (at_kitchen_content content465)
     (at_kitchen_content content364)
     (not_allergic_gluten child400)
     (not_allergic_gluten child86)
     (waiting child400 table284)
     (waiting child86 table161)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child400)
     (served child86)
    )
  )
)
