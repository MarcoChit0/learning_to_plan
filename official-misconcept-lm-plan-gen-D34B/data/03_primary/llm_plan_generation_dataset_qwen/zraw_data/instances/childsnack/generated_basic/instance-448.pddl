; child-snack task with 3 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 578513

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child175 child113 child5 - child
    bread167 bread458 bread20 - bread-portion
    content219 content141 content147 - content-portion
    tray455 - tray
    table266 table83 table309 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray455 kitchen)
     (at_kitchen_bread bread167)
     (at_kitchen_bread bread458)
     (at_kitchen_bread bread20)
     (at_kitchen_content content219)
     (at_kitchen_content content141)
     (at_kitchen_content content147)
     (not_allergic_gluten child175)
     (not_allergic_gluten child5)
     (not_allergic_gluten child113)
     (waiting child175 table309)
     (waiting child113 table309)
     (waiting child5 table83)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child175)
     (served child113)
     (served child5)
    )
  )
)
