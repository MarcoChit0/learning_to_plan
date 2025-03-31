; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 222767

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child416 child221 child331 - child
    bread38 bread181 bread164 - bread-portion
    content390 content55 content255 - content-portion
    tray429 - tray
    table372 table82 table284 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray429 kitchen)
     (at_kitchen_bread bread38)
     (at_kitchen_bread bread181)
     (at_kitchen_bread bread164)
     (at_kitchen_content content390)
     (at_kitchen_content content55)
     (at_kitchen_content content255)
     (not_allergic_gluten child331)
     (not_allergic_gluten child416)
     (not_allergic_gluten child221)
     (waiting child416 table82)
     (waiting child221 table372)
     (waiting child331 table82)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child416)
     (served child221)
     (served child331)
    )
  )
)
