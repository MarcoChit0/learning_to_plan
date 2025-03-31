; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 19516

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child476 child34 - child
    bread126 bread270 - bread-portion
    content220 content318 - content-portion
    tray397 tray156 tray479 - tray
    table242 table71 table349 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray397 kitchen)
     (at tray156 kitchen)
     (at tray479 kitchen)
     (at_kitchen_bread bread126)
     (at_kitchen_bread bread270)
     (at_kitchen_content content220)
     (at_kitchen_content content318)
     (not_allergic_gluten child476)
     (not_allergic_gluten child34)
     (waiting child476 table349)
     (waiting child34 table71)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child476)
     (served child34)
    )
  )
)
