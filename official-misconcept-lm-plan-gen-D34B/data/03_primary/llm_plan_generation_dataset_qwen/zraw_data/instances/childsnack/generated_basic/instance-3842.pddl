; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 568059

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child490 - child
    bread389 - bread-portion
    content380 - content-portion
    tray324 tray220 tray371 tray496 - tray
    table336 table181 table406 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray324 kitchen)
     (at tray220 kitchen)
     (at tray371 kitchen)
     (at tray496 kitchen)
     (at_kitchen_bread bread389)
     (at_kitchen_content content380)
     (not_allergic_gluten child490)
     (waiting child490 table181)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child490)
    )
  )
)
