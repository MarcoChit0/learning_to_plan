; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 413931

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child436 child172 - child
    bread199 bread64 - bread-portion
    content376 content424 - content-portion
    tray491 - tray
    table337 table428 table384 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray491 kitchen)
     (at_kitchen_bread bread199)
     (at_kitchen_bread bread64)
     (at_kitchen_content content376)
     (at_kitchen_content content424)
     (no_gluten_bread bread64)
     (no_gluten_content content424)
     (allergic_gluten child172)
     (not_allergic_gluten child436)
     (waiting child436 table428)
     (waiting child172 table428)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child436)
     (served child172)
    )
  )
)
