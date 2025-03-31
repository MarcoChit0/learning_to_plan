; child-snack task with 1 children and 0.2 gluten factor 
; constant factor of 1.9
; random seed: 357405

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child59 - child
    bread437 - bread-portion
    content475 - content-portion
    tray484 tray178 tray28 tray499 - tray
    table128 table361 table150 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray484 kitchen)
     (at tray178 kitchen)
     (at tray28 kitchen)
     (at tray499 kitchen)
     (at_kitchen_bread bread437)
     (at_kitchen_content content475)
     (not_allergic_gluten child59)
     (waiting child59 table361)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child59)
    )
  )
)
