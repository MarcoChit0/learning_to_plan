; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 310197

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child220 child39 - child
    bread416 bread326 - bread-portion
    content180 content420 - content-portion
    tray380 tray26 tray328 - tray
    table273 table439 table374 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray380 kitchen)
     (at tray26 kitchen)
     (at tray328 kitchen)
     (at_kitchen_bread bread416)
     (at_kitchen_bread bread326)
     (at_kitchen_content content180)
     (at_kitchen_content content420)
     (no_gluten_bread bread416)
     (no_gluten_content content420)
     (allergic_gluten child220)
     (not_allergic_gluten child39)
     (waiting child220 table439)
     (waiting child39 table273)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child220)
     (served child39)
    )
  )
)
