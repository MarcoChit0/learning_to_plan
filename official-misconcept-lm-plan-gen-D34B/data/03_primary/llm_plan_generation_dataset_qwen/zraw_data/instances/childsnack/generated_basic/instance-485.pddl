; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 61385

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child457 child206 - child
    bread346 bread314 - bread-portion
    content204 content68 - content-portion
    tray311 tray333 - tray
    table150 table452 table228 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray311 kitchen)
     (at tray333 kitchen)
     (at_kitchen_bread bread346)
     (at_kitchen_bread bread314)
     (at_kitchen_content content204)
     (at_kitchen_content content68)
     (not_allergic_gluten child206)
     (not_allergic_gluten child457)
     (waiting child457 table228)
     (waiting child206 table452)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child457)
     (served child206)
    )
  )
)
