; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 466482

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child201 child393 - child
    bread344 bread310 - bread-portion
    content494 content159 - content-portion
    tray199 tray59 tray18 - tray
    table433 table468 table359 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray199 kitchen)
     (at tray59 kitchen)
     (at tray18 kitchen)
     (at_kitchen_bread bread344)
     (at_kitchen_bread bread310)
     (at_kitchen_content content494)
     (at_kitchen_content content159)
     (not_allergic_gluten child393)
     (not_allergic_gluten child201)
     (waiting child201 table359)
     (waiting child393 table359)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child201)
     (served child393)
    )
  )
)
