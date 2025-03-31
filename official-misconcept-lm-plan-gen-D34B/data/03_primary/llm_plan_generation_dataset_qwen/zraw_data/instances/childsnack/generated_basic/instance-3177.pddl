; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 359947

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child314 child431 - child
    bread51 bread363 - bread-portion
    content135 content454 - content-portion
    tray79 tray415 - tray
    table203 table162 table36 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray79 kitchen)
     (at tray415 kitchen)
     (at_kitchen_bread bread51)
     (at_kitchen_bread bread363)
     (at_kitchen_content content135)
     (at_kitchen_content content454)
     (not_allergic_gluten child314)
     (not_allergic_gluten child431)
     (waiting child314 table162)
     (waiting child431 table203)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child314)
     (served child431)
    )
  )
)
