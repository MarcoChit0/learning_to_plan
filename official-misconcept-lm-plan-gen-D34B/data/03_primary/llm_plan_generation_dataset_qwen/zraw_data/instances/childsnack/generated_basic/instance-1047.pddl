; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 868686

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child116 child203 - child
    bread284 bread313 - bread-portion
    content125 content297 - content-portion
    tray125 tray161 tray483 - tray
    table470 table215 table83 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray125 kitchen)
     (at tray161 kitchen)
     (at tray483 kitchen)
     (at_kitchen_bread bread284)
     (at_kitchen_bread bread313)
     (at_kitchen_content content125)
     (at_kitchen_content content297)
     (not_allergic_gluten child116)
     (not_allergic_gluten child203)
     (waiting child116 table83)
     (waiting child203 table215)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child116)
     (served child203)
    )
  )
)
