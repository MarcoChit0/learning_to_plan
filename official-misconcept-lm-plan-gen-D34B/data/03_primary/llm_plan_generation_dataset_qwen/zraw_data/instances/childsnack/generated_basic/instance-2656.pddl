; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 98080

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child357 - child
    bread282 - bread-portion
    content478 - content-portion
    tray114 tray248 tray293 tray21 - tray
    table106 table370 table279 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray114 kitchen)
     (at tray248 kitchen)
     (at tray293 kitchen)
     (at tray21 kitchen)
     (at_kitchen_bread bread282)
     (at_kitchen_content content478)
     (not_allergic_gluten child357)
     (waiting child357 table106)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child357)
    )
  )
)
