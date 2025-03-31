; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 999327

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child339 - child
    bread122 - bread-portion
    content102 - content-portion
    tray347 tray6 tray149 tray32 - tray
    table433 table61 table75 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray347 kitchen)
     (at tray6 kitchen)
     (at tray149 kitchen)
     (at tray32 kitchen)
     (at_kitchen_bread bread122)
     (at_kitchen_content content102)
     (not_allergic_gluten child339)
     (waiting child339 table433)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child339)
    )
  )
)
