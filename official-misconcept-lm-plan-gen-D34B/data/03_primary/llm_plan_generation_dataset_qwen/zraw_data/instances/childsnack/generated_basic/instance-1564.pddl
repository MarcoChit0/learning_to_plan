; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 419812

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child56 child306 - child
    bread419 bread190 - bread-portion
    content218 content50 - content-portion
    tray272 tray359 tray29 tray145 - tray
    table457 table225 table63 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray272 kitchen)
     (at tray359 kitchen)
     (at tray29 kitchen)
     (at tray145 kitchen)
     (at_kitchen_bread bread419)
     (at_kitchen_bread bread190)
     (at_kitchen_content content218)
     (at_kitchen_content content50)
     (not_allergic_gluten child56)
     (not_allergic_gluten child306)
     (waiting child56 table457)
     (waiting child306 table225)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child56)
     (served child306)
    )
  )
)
