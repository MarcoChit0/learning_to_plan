; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 572087

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child402 child84 child492 - child
    bread243 bread252 bread276 - bread-portion
    content345 content431 content279 - content-portion
    tray114 - tray
    table427 table497 table359 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray114 kitchen)
     (at_kitchen_bread bread243)
     (at_kitchen_bread bread252)
     (at_kitchen_bread bread276)
     (at_kitchen_content content345)
     (at_kitchen_content content431)
     (at_kitchen_content content279)
     (not_allergic_gluten child492)
     (not_allergic_gluten child84)
     (not_allergic_gluten child402)
     (waiting child402 table427)
     (waiting child84 table427)
     (waiting child492 table497)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child402)
     (served child84)
     (served child492)
    )
  )
)
