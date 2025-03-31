; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 231589

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child95 - child
    bread78 - bread-portion
    content139 - content-portion
    tray235 tray449 - tray
    table19 table48 table108 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray235 kitchen)
     (at tray449 kitchen)
     (at_kitchen_bread bread78)
     (at_kitchen_content content139)
     (not_allergic_gluten child95)
     (waiting child95 table19)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child95)
    )
  )
)
