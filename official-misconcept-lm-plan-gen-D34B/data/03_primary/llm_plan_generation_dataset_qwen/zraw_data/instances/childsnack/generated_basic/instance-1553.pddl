; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 688453

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child220 child301 - child
    bread136 bread382 - bread-portion
    content43 content166 - content-portion
    tray374 tray182 tray416 tray38 - tray
    table349 table172 table13 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray374 kitchen)
     (at tray182 kitchen)
     (at tray416 kitchen)
     (at tray38 kitchen)
     (at_kitchen_bread bread136)
     (at_kitchen_bread bread382)
     (at_kitchen_content content43)
     (at_kitchen_content content166)
     (not_allergic_gluten child301)
     (not_allergic_gluten child220)
     (waiting child220 table13)
     (waiting child301 table349)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child220)
     (served child301)
    )
  )
)
