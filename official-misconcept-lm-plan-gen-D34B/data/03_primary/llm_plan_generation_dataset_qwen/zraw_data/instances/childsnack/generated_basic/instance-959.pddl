; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 204505

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child294 child299 child229 - child
    bread271 bread252 bread440 - bread-portion
    content344 content145 content434 - content-portion
    tray494 tray443 tray111 tray487 - tray
    table71 table14 table127 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray494 kitchen)
     (at tray443 kitchen)
     (at tray111 kitchen)
     (at tray487 kitchen)
     (at_kitchen_bread bread271)
     (at_kitchen_bread bread252)
     (at_kitchen_bread bread440)
     (at_kitchen_content content344)
     (at_kitchen_content content145)
     (at_kitchen_content content434)
     (not_allergic_gluten child294)
     (not_allergic_gluten child299)
     (not_allergic_gluten child229)
     (waiting child294 table127)
     (waiting child299 table14)
     (waiting child229 table127)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child294)
     (served child299)
     (served child229)
    )
  )
)
