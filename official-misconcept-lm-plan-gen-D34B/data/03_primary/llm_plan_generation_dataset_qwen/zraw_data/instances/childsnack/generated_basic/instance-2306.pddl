; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 267617

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child92 - child
    bread375 - bread-portion
    content188 - content-portion
    tray443 - tray
    table461 table447 table27 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray443 kitchen)
     (at_kitchen_bread bread375)
     (at_kitchen_content content188)
     (not_allergic_gluten child92)
     (waiting child92 table461)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child92)
    )
  )
)
