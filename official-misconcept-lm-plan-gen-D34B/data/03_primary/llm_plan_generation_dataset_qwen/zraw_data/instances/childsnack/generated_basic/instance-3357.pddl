; child-snack task with 2 children and 0.6 gluten factor 
; constant factor of 1.3
; random seed: 957143

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child38 child298 - child
    bread218 bread33 - bread-portion
    content340 content209 - content-portion
    tray346 tray238 - tray
    table302 table324 table128 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray346 kitchen)
     (at tray238 kitchen)
     (at_kitchen_bread bread218)
     (at_kitchen_bread bread33)
     (at_kitchen_content content340)
     (at_kitchen_content content209)
     (no_gluten_bread bread218)
     (no_gluten_content content340)
     (allergic_gluten child38)
     (not_allergic_gluten child298)
     (waiting child38 table128)
     (waiting child298 table302)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child38)
     (served child298)
    )
  )
)
