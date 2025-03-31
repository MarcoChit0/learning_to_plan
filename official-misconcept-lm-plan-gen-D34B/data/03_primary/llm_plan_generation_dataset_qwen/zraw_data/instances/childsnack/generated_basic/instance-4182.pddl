; child-snack task with 2 children and 0.28 gluten factor 
; constant factor of 1.7
; random seed: 802485

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child409 child469 - child
    bread297 bread87 - bread-portion
    content54 content15 - content-portion
    tray294 - tray
    table337 table152 table345 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray294 kitchen)
     (at_kitchen_bread bread297)
     (at_kitchen_bread bread87)
     (at_kitchen_content content54)
     (at_kitchen_content content15)
     (not_allergic_gluten child409)
     (not_allergic_gluten child469)
     (waiting child409 table345)
     (waiting child469 table337)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child409)
     (served child469)
    )
  )
)
