; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 189998

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child178 child17 - child
    bread291 bread462 - bread-portion
    content368 content335 - content-portion
    tray348 tray423 - tray
    table43 table468 table180 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray348 kitchen)
     (at tray423 kitchen)
     (at_kitchen_bread bread291)
     (at_kitchen_bread bread462)
     (at_kitchen_content content368)
     (at_kitchen_content content335)
     (not_allergic_gluten child178)
     (not_allergic_gluten child17)
     (waiting child178 table43)
     (waiting child17 table180)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child178)
     (served child17)
    )
  )
)
