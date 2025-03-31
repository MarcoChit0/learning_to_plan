; child-snack task with 2 children and 0.56 gluten factor 
; constant factor of 1.3
; random seed: 763024

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child150 child26 - child
    bread199 bread471 - bread-portion
    content270 content249 - content-portion
    tray367 tray498 - tray
    table10 table324 table242 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray367 kitchen)
     (at tray498 kitchen)
     (at_kitchen_bread bread199)
     (at_kitchen_bread bread471)
     (at_kitchen_content content270)
     (at_kitchen_content content249)
     (no_gluten_bread bread471)
     (no_gluten_content content249)
     (allergic_gluten child26)
     (not_allergic_gluten child150)
     (waiting child150 table242)
     (waiting child26 table10)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child150)
     (served child26)
    )
  )
)
