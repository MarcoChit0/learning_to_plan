; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 199041

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child214 child115 - child
    bread34 bread161 - bread-portion
    content243 content216 - content-portion
    tray229 - tray
    table248 table53 table63 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray229 kitchen)
     (at_kitchen_bread bread34)
     (at_kitchen_bread bread161)
     (at_kitchen_content content243)
     (at_kitchen_content content216)
     (no_gluten_bread bread161)
     (no_gluten_content content216)
     (allergic_gluten child115)
     (not_allergic_gluten child214)
     (waiting child214 table53)
     (waiting child115 table53)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child214)
     (served child115)
    )
  )
)
