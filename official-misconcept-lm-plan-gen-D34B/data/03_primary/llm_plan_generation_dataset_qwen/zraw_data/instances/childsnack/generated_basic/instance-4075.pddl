; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 138047

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child34 - child
    bread258 - bread-portion
    content59 - content-portion
    tray20 tray207 tray474 - tray
    table96 table150 table194 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray20 kitchen)
     (at tray207 kitchen)
     (at tray474 kitchen)
     (at_kitchen_bread bread258)
     (at_kitchen_content content59)
     (not_allergic_gluten child34)
     (waiting child34 table194)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child34)
    )
  )
)
