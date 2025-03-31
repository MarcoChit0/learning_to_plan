; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 172386

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child415 child239 - child
    bread126 bread219 - bread-portion
    content98 content24 - content-portion
    tray39 tray151 tray417 - tray
    table358 table2 table14 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray39 kitchen)
     (at tray151 kitchen)
     (at tray417 kitchen)
     (at_kitchen_bread bread126)
     (at_kitchen_bread bread219)
     (at_kitchen_content content98)
     (at_kitchen_content content24)
     (no_gluten_bread bread126)
     (no_gluten_content content98)
     (allergic_gluten child415)
     (not_allergic_gluten child239)
     (waiting child415 table358)
     (waiting child239 table14)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child415)
     (served child239)
    )
  )
)
