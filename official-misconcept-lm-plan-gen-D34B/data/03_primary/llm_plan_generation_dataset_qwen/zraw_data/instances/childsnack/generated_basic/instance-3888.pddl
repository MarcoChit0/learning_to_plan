; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 493489

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child270 - child
    bread235 - bread-portion
    content300 - content-portion
    tray279 tray213 - tray
    table257 table339 table391 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray279 kitchen)
     (at tray213 kitchen)
     (at_kitchen_bread bread235)
     (at_kitchen_content content300)
     (not_allergic_gluten child270)
     (waiting child270 table391)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child270)
    )
  )
)
