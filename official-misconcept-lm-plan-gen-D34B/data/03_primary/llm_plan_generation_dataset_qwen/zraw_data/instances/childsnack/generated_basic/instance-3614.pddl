; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 104599

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child225 child345 - child
    bread238 bread74 - bread-portion
    content46 content345 - content-portion
    tray16 tray423 tray252 tray468 - tray
    table223 table21 table324 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray16 kitchen)
     (at tray423 kitchen)
     (at tray252 kitchen)
     (at tray468 kitchen)
     (at_kitchen_bread bread238)
     (at_kitchen_bread bread74)
     (at_kitchen_content content46)
     (at_kitchen_content content345)
     (no_gluten_bread bread74)
     (no_gluten_content content46)
     (allergic_gluten child345)
     (not_allergic_gluten child225)
     (waiting child225 table21)
     (waiting child345 table21)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child225)
     (served child345)
    )
  )
)
