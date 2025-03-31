; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 358204

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child20 child68 - child
    bread113 bread13 - bread-portion
    content317 content403 - content-portion
    tray360 - tray
    table227 table79 table390 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray360 kitchen)
     (at_kitchen_bread bread113)
     (at_kitchen_bread bread13)
     (at_kitchen_content content317)
     (at_kitchen_content content403)
     (no_gluten_bread bread113)
     (no_gluten_content content317)
     (allergic_gluten child20)
     (not_allergic_gluten child68)
     (waiting child20 table227)
     (waiting child68 table79)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child20)
     (served child68)
    )
  )
)
