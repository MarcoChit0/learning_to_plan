; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 679410

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child150 child493 - child
    bread32 bread167 - bread-portion
    content342 content402 - content-portion
    tray343 tray420 tray333 tray461 - tray
    table277 table464 table446 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray343 kitchen)
     (at tray420 kitchen)
     (at tray333 kitchen)
     (at tray461 kitchen)
     (at_kitchen_bread bread32)
     (at_kitchen_bread bread167)
     (at_kitchen_content content342)
     (at_kitchen_content content402)
     (not_allergic_gluten child493)
     (not_allergic_gluten child150)
     (waiting child150 table464)
     (waiting child493 table446)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child150)
     (served child493)
    )
  )
)
