; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 155127

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child215 child250 - child
    bread276 bread112 - bread-portion
    content254 content241 - content-portion
    tray447 tray334 tray78 - tray
    table14 table157 table420 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray447 kitchen)
     (at tray334 kitchen)
     (at tray78 kitchen)
     (at_kitchen_bread bread276)
     (at_kitchen_bread bread112)
     (at_kitchen_content content254)
     (at_kitchen_content content241)
     (not_allergic_gluten child250)
     (not_allergic_gluten child215)
     (waiting child215 table14)
     (waiting child250 table157)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child215)
     (served child250)
    )
  )
)
