; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 26293

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child481 - child
    bread461 - bread-portion
    content381 - content-portion
    tray442 tray319 tray402 - tray
    table260 table370 table237 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray442 kitchen)
     (at tray319 kitchen)
     (at tray402 kitchen)
     (at_kitchen_bread bread461)
     (at_kitchen_content content381)
     (not_allergic_gluten child481)
     (waiting child481 table260)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child481)
    )
  )
)
