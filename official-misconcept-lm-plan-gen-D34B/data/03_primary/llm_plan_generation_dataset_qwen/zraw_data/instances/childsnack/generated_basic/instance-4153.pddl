; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 867157

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child113 child244 - child
    bread288 bread225 - bread-portion
    content46 content390 - content-portion
    tray308 tray278 tray29 - tray
    table24 table9 table141 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray308 kitchen)
     (at tray278 kitchen)
     (at tray29 kitchen)
     (at_kitchen_bread bread288)
     (at_kitchen_bread bread225)
     (at_kitchen_content content46)
     (at_kitchen_content content390)
     (not_allergic_gluten child113)
     (not_allergic_gluten child244)
     (waiting child113 table24)
     (waiting child244 table24)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child113)
     (served child244)
    )
  )
)
