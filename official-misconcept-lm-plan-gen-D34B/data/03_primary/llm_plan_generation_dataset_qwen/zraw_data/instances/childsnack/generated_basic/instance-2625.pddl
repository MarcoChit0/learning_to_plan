; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 93684

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child491 - child
    bread35 - bread-portion
    content301 - content-portion
    tray110 tray383 tray423 - tray
    table221 table474 table489 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray110 kitchen)
     (at tray383 kitchen)
     (at tray423 kitchen)
     (at_kitchen_bread bread35)
     (at_kitchen_content content301)
     (not_allergic_gluten child491)
     (waiting child491 table221)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child491)
    )
  )
)
