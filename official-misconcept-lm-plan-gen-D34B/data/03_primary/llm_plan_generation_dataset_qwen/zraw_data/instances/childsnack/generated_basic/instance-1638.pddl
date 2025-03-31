; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 756827

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child307 - child
    bread246 - bread-portion
    content275 - content-portion
    tray396 tray234 tray371 tray454 - tray
    table328 table265 table410 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray396 kitchen)
     (at tray234 kitchen)
     (at tray371 kitchen)
     (at tray454 kitchen)
     (at_kitchen_bread bread246)
     (at_kitchen_content content275)
     (not_allergic_gluten child307)
     (waiting child307 table265)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child307)
    )
  )
)
