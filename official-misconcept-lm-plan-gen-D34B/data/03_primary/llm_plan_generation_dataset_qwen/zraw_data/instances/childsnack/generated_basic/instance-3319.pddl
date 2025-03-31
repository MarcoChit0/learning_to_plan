; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 95874

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child122 child262 - child
    bread21 bread3 - bread-portion
    content25 content30 - content-portion
    tray197 tray129 tray192 tray301 - tray
    table109 table198 table341 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray197 kitchen)
     (at tray129 kitchen)
     (at tray192 kitchen)
     (at tray301 kitchen)
     (at_kitchen_bread bread21)
     (at_kitchen_bread bread3)
     (at_kitchen_content content25)
     (at_kitchen_content content30)
     (no_gluten_bread bread21)
     (no_gluten_content content30)
     (allergic_gluten child122)
     (not_allergic_gluten child262)
     (waiting child122 table341)
     (waiting child262 table109)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child122)
     (served child262)
    )
  )
)
