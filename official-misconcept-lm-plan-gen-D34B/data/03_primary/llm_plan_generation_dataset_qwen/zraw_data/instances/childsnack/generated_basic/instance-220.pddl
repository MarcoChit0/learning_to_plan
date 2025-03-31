; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 947457

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child259 child250 - child
    bread216 bread55 - bread-portion
    content231 content121 - content-portion
    tray343 tray64 tray356 - tray
    table126 table498 table114 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray343 kitchen)
     (at tray64 kitchen)
     (at tray356 kitchen)
     (at_kitchen_bread bread216)
     (at_kitchen_bread bread55)
     (at_kitchen_content content231)
     (at_kitchen_content content121)
     (not_allergic_gluten child250)
     (not_allergic_gluten child259)
     (waiting child259 table126)
     (waiting child250 table114)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child259)
     (served child250)
    )
  )
)
