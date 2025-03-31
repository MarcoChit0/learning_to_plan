; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 476345

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child238 - child
    bread246 - bread-portion
    content162 - content-portion
    tray153 tray15 tray140 tray322 - tray
    table496 table12 table48 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray153 kitchen)
     (at tray15 kitchen)
     (at tray140 kitchen)
     (at tray322 kitchen)
     (at_kitchen_bread bread246)
     (at_kitchen_content content162)
     (not_allergic_gluten child238)
     (waiting child238 table496)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child238)
    )
  )
)
