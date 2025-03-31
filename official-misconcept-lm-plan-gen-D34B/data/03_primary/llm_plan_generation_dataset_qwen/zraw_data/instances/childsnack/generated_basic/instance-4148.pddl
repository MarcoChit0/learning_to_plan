; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 797955

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child173 child24 - child
    bread270 bread204 - bread-portion
    content353 content192 - content-portion
    tray261 tray492 tray263 - tray
    table171 table394 table300 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray261 kitchen)
     (at tray492 kitchen)
     (at tray263 kitchen)
     (at_kitchen_bread bread270)
     (at_kitchen_bread bread204)
     (at_kitchen_content content353)
     (at_kitchen_content content192)
     (not_allergic_gluten child24)
     (not_allergic_gluten child173)
     (waiting child173 table300)
     (waiting child24 table171)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child173)
     (served child24)
    )
  )
)
