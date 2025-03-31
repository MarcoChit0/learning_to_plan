; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 644367

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child5 child306 - child
    bread176 bread272 - bread-portion
    content476 content496 - content-portion
    tray420 tray381 - tray
    table427 table174 table301 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray420 kitchen)
     (at tray381 kitchen)
     (at_kitchen_bread bread176)
     (at_kitchen_bread bread272)
     (at_kitchen_content content476)
     (at_kitchen_content content496)
     (not_allergic_gluten child306)
     (not_allergic_gluten child5)
     (waiting child5 table301)
     (waiting child306 table174)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child5)
     (served child306)
    )
  )
)
