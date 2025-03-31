; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 253915

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child95 child44 - child
    bread474 bread49 - bread-portion
    content394 content303 - content-portion
    tray6 tray289 tray367 - tray
    table184 table418 table44 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray6 kitchen)
     (at tray289 kitchen)
     (at tray367 kitchen)
     (at_kitchen_bread bread474)
     (at_kitchen_bread bread49)
     (at_kitchen_content content394)
     (at_kitchen_content content303)
     (not_allergic_gluten child44)
     (not_allergic_gluten child95)
     (waiting child95 table184)
     (waiting child44 table184)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child95)
     (served child44)
    )
  )
)
