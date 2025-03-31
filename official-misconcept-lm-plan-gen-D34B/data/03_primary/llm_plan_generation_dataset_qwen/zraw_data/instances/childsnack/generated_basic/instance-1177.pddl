; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 803968

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child148 child345 - child
    bread76 bread114 - bread-portion
    content334 content366 - content-portion
    tray18 tray450 tray458 - tray
    table187 table383 table286 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray18 kitchen)
     (at tray450 kitchen)
     (at tray458 kitchen)
     (at_kitchen_bread bread76)
     (at_kitchen_bread bread114)
     (at_kitchen_content content334)
     (at_kitchen_content content366)
     (no_gluten_bread bread114)
     (no_gluten_content content334)
     (allergic_gluten child148)
     (not_allergic_gluten child345)
     (waiting child148 table383)
     (waiting child345 table187)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child148)
     (served child345)
    )
  )
)
