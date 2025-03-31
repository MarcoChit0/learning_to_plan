; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 30769

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child56 - child
    bread59 - bread-portion
    content223 - content-portion
    tray438 tray37 tray11 tray347 - tray
    table443 table488 table221 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray438 kitchen)
     (at tray37 kitchen)
     (at tray11 kitchen)
     (at tray347 kitchen)
     (at_kitchen_bread bread59)
     (at_kitchen_content content223)
     (not_allergic_gluten child56)
     (waiting child56 table221)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child56)
    )
  )
)
