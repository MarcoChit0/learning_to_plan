; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 710118

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child275 - child
    bread423 - bread-portion
    content77 - content-portion
    tray271 tray353 tray323 tray421 - tray
    table452 table478 table289 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray271 kitchen)
     (at tray353 kitchen)
     (at tray323 kitchen)
     (at tray421 kitchen)
     (at_kitchen_bread bread423)
     (at_kitchen_content content77)
     (not_allergic_gluten child275)
     (waiting child275 table452)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child275)
    )
  )
)
