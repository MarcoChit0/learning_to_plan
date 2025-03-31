; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 351907

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child196 - child
    bread463 - bread-portion
    content394 - content-portion
    tray342 tray306 tray366 - tray
    table253 table159 table66 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray342 kitchen)
     (at tray306 kitchen)
     (at tray366 kitchen)
     (at_kitchen_bread bread463)
     (at_kitchen_content content394)
     (not_allergic_gluten child196)
     (waiting child196 table159)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child196)
    )
  )
)
