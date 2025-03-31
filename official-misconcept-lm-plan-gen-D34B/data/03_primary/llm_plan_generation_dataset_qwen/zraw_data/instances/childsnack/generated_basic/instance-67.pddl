; child-snack task with 2 children and 0.36 gluten factor 
; constant factor of 1.9
; random seed: 531634

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child341 child24 - child
    bread215 bread119 - bread-portion
    content314 content73 - content-portion
    tray390 tray32 - tray
    table191 table101 table271 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray390 kitchen)
     (at tray32 kitchen)
     (at_kitchen_bread bread215)
     (at_kitchen_bread bread119)
     (at_kitchen_content content314)
     (at_kitchen_content content73)
     (not_allergic_gluten child24)
     (not_allergic_gluten child341)
     (waiting child341 table271)
     (waiting child24 table271)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child341)
     (served child24)
    )
  )
)
