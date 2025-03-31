; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 825495

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child427 child73 - child
    bread324 bread77 - bread-portion
    content363 content138 - content-portion
    tray445 tray188 - tray
    table98 table268 table218 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray445 kitchen)
     (at tray188 kitchen)
     (at_kitchen_bread bread324)
     (at_kitchen_bread bread77)
     (at_kitchen_content content363)
     (at_kitchen_content content138)
     (no_gluten_bread bread324)
     (no_gluten_content content363)
     (allergic_gluten child427)
     (not_allergic_gluten child73)
     (waiting child427 table218)
     (waiting child73 table98)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child427)
     (served child73)
    )
  )
)
