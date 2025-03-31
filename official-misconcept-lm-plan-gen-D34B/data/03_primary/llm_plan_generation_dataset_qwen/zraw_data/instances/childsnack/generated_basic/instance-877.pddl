; child-snack task with 2 children and 0.59 gluten factor 
; constant factor of 1.9
; random seed: 37122

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child36 child378 - child
    bread430 bread80 - bread-portion
    content83 content241 - content-portion
    tray277 tray488 - tray
    table222 table430 table307 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray277 kitchen)
     (at tray488 kitchen)
     (at_kitchen_bread bread430)
     (at_kitchen_bread bread80)
     (at_kitchen_content content83)
     (at_kitchen_content content241)
     (no_gluten_bread bread430)
     (no_gluten_content content83)
     (allergic_gluten child36)
     (not_allergic_gluten child378)
     (waiting child36 table307)
     (waiting child378 table222)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child36)
     (served child378)
    )
  )
)
