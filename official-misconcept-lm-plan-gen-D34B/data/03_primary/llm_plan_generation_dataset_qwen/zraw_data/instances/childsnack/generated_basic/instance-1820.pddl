; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 966561

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child413 child423 - child
    bread147 bread42 - bread-portion
    content405 content369 - content-portion
    tray323 - tray
    table273 table39 table133 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray323 kitchen)
     (at_kitchen_bread bread147)
     (at_kitchen_bread bread42)
     (at_kitchen_content content405)
     (at_kitchen_content content369)
     (no_gluten_bread bread147)
     (no_gluten_content content405)
     (allergic_gluten child423)
     (not_allergic_gluten child413)
     (waiting child413 table133)
     (waiting child423 table39)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child413)
     (served child423)
    )
  )
)
