; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 709785

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child186 child82 - child
    bread267 bread262 - bread-portion
    content254 content152 - content-portion
    tray132 tray493 tray433 - tray
    table363 table287 table126 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray132 kitchen)
     (at tray493 kitchen)
     (at tray433 kitchen)
     (at_kitchen_bread bread267)
     (at_kitchen_bread bread262)
     (at_kitchen_content content254)
     (at_kitchen_content content152)
     (no_gluten_bread bread262)
     (no_gluten_content content152)
     (allergic_gluten child82)
     (not_allergic_gluten child186)
     (waiting child186 table363)
     (waiting child82 table287)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child186)
     (served child82)
    )
  )
)
