; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 836129

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child450 - child
    bread422 - bread-portion
    content383 - content-portion
    tray33 tray408 tray493 tray115 - tray
    table314 table114 table155 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray33 kitchen)
     (at tray408 kitchen)
     (at tray493 kitchen)
     (at tray115 kitchen)
     (at_kitchen_bread bread422)
     (at_kitchen_content content383)
     (not_allergic_gluten child450)
     (waiting child450 table314)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child450)
    )
  )
)
