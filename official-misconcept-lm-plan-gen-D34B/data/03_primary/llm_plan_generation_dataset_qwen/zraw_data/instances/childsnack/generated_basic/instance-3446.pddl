; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 406003

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child331 - child
    bread157 - bread-portion
    content54 - content-portion
    tray495 tray423 tray18 tray0 - tray
    table343 table319 table147 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray495 kitchen)
     (at tray423 kitchen)
     (at tray18 kitchen)
     (at tray0 kitchen)
     (at_kitchen_bread bread157)
     (at_kitchen_content content54)
     (not_allergic_gluten child331)
     (waiting child331 table319)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child331)
    )
  )
)
