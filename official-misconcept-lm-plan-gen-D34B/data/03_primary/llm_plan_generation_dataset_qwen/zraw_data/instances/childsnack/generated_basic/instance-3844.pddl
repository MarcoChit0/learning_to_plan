; child-snack task with 1 children and 0.47 gluten factor 
; constant factor of 1.7
; random seed: 113024

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child331 - child
    bread322 - bread-portion
    content364 - content-portion
    tray107 tray385 tray322 tray147 - tray
    table19 table157 table392 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray107 kitchen)
     (at tray385 kitchen)
     (at tray322 kitchen)
     (at tray147 kitchen)
     (at_kitchen_bread bread322)
     (at_kitchen_content content364)
     (not_allergic_gluten child331)
     (waiting child331 table392)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child331)
    )
  )
)
