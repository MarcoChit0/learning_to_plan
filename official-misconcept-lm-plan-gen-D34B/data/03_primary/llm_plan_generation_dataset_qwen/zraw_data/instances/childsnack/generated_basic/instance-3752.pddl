; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 362442

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child284 - child
    bread323 - bread-portion
    content145 - content-portion
    tray262 tray463 tray159 - tray
    table131 table449 table430 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray262 kitchen)
     (at tray463 kitchen)
     (at tray159 kitchen)
     (at_kitchen_bread bread323)
     (at_kitchen_content content145)
     (not_allergic_gluten child284)
     (waiting child284 table430)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child284)
    )
  )
)
