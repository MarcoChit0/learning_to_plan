; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 209286

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child213 child198 - child
    bread42 bread112 - bread-portion
    content363 content187 - content-portion
    tray480 tray425 tray361 - tray
    table5 table88 table31 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray480 kitchen)
     (at tray425 kitchen)
     (at tray361 kitchen)
     (at_kitchen_bread bread42)
     (at_kitchen_bread bread112)
     (at_kitchen_content content363)
     (at_kitchen_content content187)
     (no_gluten_bread bread112)
     (no_gluten_content content363)
     (allergic_gluten child213)
     (not_allergic_gluten child198)
     (waiting child213 table5)
     (waiting child198 table88)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child213)
     (served child198)
    )
  )
)
