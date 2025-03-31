; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 473435

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child24 child106 - child
    bread491 bread193 - bread-portion
    content479 content426 - content-portion
    tray99 tray181 tray275 tray101 - tray
    table283 table397 table277 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray99 kitchen)
     (at tray181 kitchen)
     (at tray275 kitchen)
     (at tray101 kitchen)
     (at_kitchen_bread bread491)
     (at_kitchen_bread bread193)
     (at_kitchen_content content479)
     (at_kitchen_content content426)
     (not_allergic_gluten child106)
     (not_allergic_gluten child24)
     (waiting child24 table397)
     (waiting child106 table277)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child24)
     (served child106)
    )
  )
)
