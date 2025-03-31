; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 196168

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child209 child14 - child
    bread415 bread55 - bread-portion
    content257 content5 - content-portion
    tray483 tray408 tray74 - tray
    table51 table199 table356 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray483 kitchen)
     (at tray408 kitchen)
     (at tray74 kitchen)
     (at_kitchen_bread bread415)
     (at_kitchen_bread bread55)
     (at_kitchen_content content257)
     (at_kitchen_content content5)
     (not_allergic_gluten child209)
     (not_allergic_gluten child14)
     (waiting child209 table199)
     (waiting child14 table51)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child209)
     (served child14)
    )
  )
)
