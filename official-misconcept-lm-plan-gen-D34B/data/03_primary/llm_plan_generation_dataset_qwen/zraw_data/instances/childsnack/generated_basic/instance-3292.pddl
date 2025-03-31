; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 188201

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child239 child234 - child
    bread446 bread339 - bread-portion
    content231 content399 - content-portion
    tray414 tray166 tray335 tray86 - tray
    table15 table462 table348 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray414 kitchen)
     (at tray166 kitchen)
     (at tray335 kitchen)
     (at tray86 kitchen)
     (at_kitchen_bread bread446)
     (at_kitchen_bread bread339)
     (at_kitchen_content content231)
     (at_kitchen_content content399)
     (no_gluten_bread bread446)
     (no_gluten_content content399)
     (allergic_gluten child239)
     (not_allergic_gluten child234)
     (waiting child239 table348)
     (waiting child234 table462)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child239)
     (served child234)
    )
  )
)
