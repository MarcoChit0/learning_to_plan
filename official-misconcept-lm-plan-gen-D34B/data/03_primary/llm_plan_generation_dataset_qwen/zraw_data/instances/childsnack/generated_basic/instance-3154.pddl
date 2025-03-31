; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 922788

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child338 child215 - child
    bread279 bread354 - bread-portion
    content100 content160 - content-portion
    tray261 tray217 - tray
    table287 table316 table202 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray261 kitchen)
     (at tray217 kitchen)
     (at_kitchen_bread bread279)
     (at_kitchen_bread bread354)
     (at_kitchen_content content100)
     (at_kitchen_content content160)
     (not_allergic_gluten child338)
     (not_allergic_gluten child215)
     (waiting child338 table202)
     (waiting child215 table287)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child338)
     (served child215)
    )
  )
)
