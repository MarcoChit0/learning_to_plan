; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 823161

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child249 child116 - child
    bread374 bread18 - bread-portion
    content350 content281 - content-portion
    tray135 - tray
    table359 table262 table236 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray135 kitchen)
     (at_kitchen_bread bread374)
     (at_kitchen_bread bread18)
     (at_kitchen_content content350)
     (at_kitchen_content content281)
     (no_gluten_bread bread374)
     (no_gluten_content content281)
     (allergic_gluten child116)
     (not_allergic_gluten child249)
     (waiting child249 table262)
     (waiting child116 table262)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child249)
     (served child116)
    )
  )
)
