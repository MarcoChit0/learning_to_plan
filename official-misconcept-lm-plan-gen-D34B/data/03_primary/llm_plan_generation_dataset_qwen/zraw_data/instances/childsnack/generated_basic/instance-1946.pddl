; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 875704

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child199 - child
    bread332 - bread-portion
    content448 - content-portion
    tray61 tray33 tray191 - tray
    table222 table167 table74 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray61 kitchen)
     (at tray33 kitchen)
     (at tray191 kitchen)
     (at_kitchen_bread bread332)
     (at_kitchen_content content448)
     (not_allergic_gluten child199)
     (waiting child199 table222)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child199)
    )
  )
)
