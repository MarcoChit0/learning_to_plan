; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 566465

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child132 - child
    bread449 - bread-portion
    content288 - content-portion
    tray285 tray463 tray480 - tray
    table40 table93 table356 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray285 kitchen)
     (at tray463 kitchen)
     (at tray480 kitchen)
     (at_kitchen_bread bread449)
     (at_kitchen_content content288)
     (not_allergic_gluten child132)
     (waiting child132 table93)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child132)
    )
  )
)
