; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 566399

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child132 child435 - child
    bread442 bread427 - bread-portion
    content152 content491 - content-portion
    tray199 tray153 tray88 - tray
    table399 table48 table121 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray199 kitchen)
     (at tray153 kitchen)
     (at tray88 kitchen)
     (at_kitchen_bread bread442)
     (at_kitchen_bread bread427)
     (at_kitchen_content content152)
     (at_kitchen_content content491)
     (not_allergic_gluten child132)
     (not_allergic_gluten child435)
     (waiting child132 table121)
     (waiting child435 table121)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child132)
     (served child435)
    )
  )
)
