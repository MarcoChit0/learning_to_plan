; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 927698

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child219 child136 - child
    bread197 bread52 - bread-portion
    content261 content191 - content-portion
    tray134 tray452 tray110 tray469 - tray
    table78 table420 table380 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray134 kitchen)
     (at tray452 kitchen)
     (at tray110 kitchen)
     (at tray469 kitchen)
     (at_kitchen_bread bread197)
     (at_kitchen_bread bread52)
     (at_kitchen_content content261)
     (at_kitchen_content content191)
     (not_allergic_gluten child219)
     (not_allergic_gluten child136)
     (waiting child219 table420)
     (waiting child136 table420)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child219)
     (served child136)
    )
  )
)
