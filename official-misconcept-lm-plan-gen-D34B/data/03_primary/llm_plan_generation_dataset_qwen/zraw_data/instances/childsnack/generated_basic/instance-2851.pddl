; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 901283

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child33 - child
    bread346 - bread-portion
    content348 - content-portion
    tray459 tray383 - tray
    table212 table423 table204 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray459 kitchen)
     (at tray383 kitchen)
     (at_kitchen_bread bread346)
     (at_kitchen_content content348)
     (not_allergic_gluten child33)
     (waiting child33 table212)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child33)
    )
  )
)
