; child-snack task with 1 children and 0.48 gluten factor 
; constant factor of 1.7
; random seed: 382476

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child163 - child
    bread322 - bread-portion
    content170 - content-portion
    tray185 tray25 tray369 - tray
    table474 table211 table448 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray185 kitchen)
     (at tray25 kitchen)
     (at tray369 kitchen)
     (at_kitchen_bread bread322)
     (at_kitchen_content content170)
     (not_allergic_gluten child163)
     (waiting child163 table211)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child163)
    )
  )
)
