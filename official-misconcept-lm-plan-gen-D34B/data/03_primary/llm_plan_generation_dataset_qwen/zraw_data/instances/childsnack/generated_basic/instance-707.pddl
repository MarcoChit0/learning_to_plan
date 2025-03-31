; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 166187

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child199 - child
    bread42 - bread-portion
    content413 - content-portion
    tray17 tray322 tray271 tray499 - tray
    table295 table156 table14 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray17 kitchen)
     (at tray322 kitchen)
     (at tray271 kitchen)
     (at tray499 kitchen)
     (at_kitchen_bread bread42)
     (at_kitchen_content content413)
     (not_allergic_gluten child199)
     (waiting child199 table295)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child199)
    )
  )
)
