; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 870435

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child108 child306 - child
    bread36 bread375 - bread-portion
    content434 content379 - content-portion
    tray356 tray368 - tray
    table219 table260 table359 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray356 kitchen)
     (at tray368 kitchen)
     (at_kitchen_bread bread36)
     (at_kitchen_bread bread375)
     (at_kitchen_content content434)
     (at_kitchen_content content379)
     (no_gluten_bread bread36)
     (no_gluten_content content379)
     (allergic_gluten child306)
     (not_allergic_gluten child108)
     (waiting child108 table359)
     (waiting child306 table260)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child108)
     (served child306)
    )
  )
)
