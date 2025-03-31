; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 171889

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child65 child352 - child
    bread10 bread89 - bread-portion
    content230 content146 - content-portion
    tray45 - tray
    table46 table421 table176 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray45 kitchen)
     (at_kitchen_bread bread10)
     (at_kitchen_bread bread89)
     (at_kitchen_content content230)
     (at_kitchen_content content146)
     (no_gluten_bread bread89)
     (no_gluten_content content146)
     (allergic_gluten child352)
     (not_allergic_gluten child65)
     (waiting child65 table421)
     (waiting child352 table46)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child65)
     (served child352)
    )
  )
)
