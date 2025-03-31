; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 21550

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child72 child319 - child
    bread318 bread445 - bread-portion
    content56 content427 - content-portion
    tray455 tray289 tray343 - tray
    table38 table480 table123 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray455 kitchen)
     (at tray289 kitchen)
     (at tray343 kitchen)
     (at_kitchen_bread bread318)
     (at_kitchen_bread bread445)
     (at_kitchen_content content56)
     (at_kitchen_content content427)
     (no_gluten_bread bread318)
     (no_gluten_content content427)
     (allergic_gluten child72)
     (not_allergic_gluten child319)
     (waiting child72 table480)
     (waiting child319 table480)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child72)
     (served child319)
    )
  )
)
