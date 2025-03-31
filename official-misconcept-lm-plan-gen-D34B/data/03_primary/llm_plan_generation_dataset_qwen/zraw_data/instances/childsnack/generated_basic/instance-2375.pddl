; child-snack task with 2 children and 0.38 gluten factor 
; constant factor of 1.9
; random seed: 86700

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child218 child475 - child
    bread88 bread46 - bread-portion
    content88 content23 - content-portion
    tray107 - tray
    table460 table155 table86 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray107 kitchen)
     (at_kitchen_bread bread88)
     (at_kitchen_bread bread46)
     (at_kitchen_content content88)
     (at_kitchen_content content23)
     (not_allergic_gluten child218)
     (not_allergic_gluten child475)
     (waiting child218 table155)
     (waiting child475 table86)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child218)
     (served child475)
    )
  )
)
