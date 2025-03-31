; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 817647

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child343 child275 - child
    bread410 bread320 - bread-portion
    content464 content328 - content-portion
    tray386 - tray
    table14 table347 table217 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray386 kitchen)
     (at_kitchen_bread bread410)
     (at_kitchen_bread bread320)
     (at_kitchen_content content464)
     (at_kitchen_content content328)
     (no_gluten_bread bread410)
     (no_gluten_content content328)
     (allergic_gluten child343)
     (not_allergic_gluten child275)
     (waiting child343 table14)
     (waiting child275 table14)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child343)
     (served child275)
    )
  )
)
