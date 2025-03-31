; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 997464

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child345 child262 - child
    bread62 bread304 - bread-portion
    content28 content114 - content-portion
    tray200 - tray
    table421 table245 table32 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray200 kitchen)
     (at_kitchen_bread bread62)
     (at_kitchen_bread bread304)
     (at_kitchen_content content28)
     (at_kitchen_content content114)
     (not_allergic_gluten child262)
     (not_allergic_gluten child345)
     (waiting child345 table245)
     (waiting child262 table245)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child345)
     (served child262)
    )
  )
)
