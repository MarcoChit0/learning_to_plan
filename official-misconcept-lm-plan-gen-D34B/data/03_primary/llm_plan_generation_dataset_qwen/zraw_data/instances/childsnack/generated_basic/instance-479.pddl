; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 271274

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child24 child218 - child
    bread194 bread157 - bread-portion
    content187 content79 - content-portion
    tray135 tray499 - tray
    table461 table410 table23 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray135 kitchen)
     (at tray499 kitchen)
     (at_kitchen_bread bread194)
     (at_kitchen_bread bread157)
     (at_kitchen_content content187)
     (at_kitchen_content content79)
     (not_allergic_gluten child218)
     (not_allergic_gluten child24)
     (waiting child24 table461)
     (waiting child218 table23)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child24)
     (served child218)
    )
  )
)
