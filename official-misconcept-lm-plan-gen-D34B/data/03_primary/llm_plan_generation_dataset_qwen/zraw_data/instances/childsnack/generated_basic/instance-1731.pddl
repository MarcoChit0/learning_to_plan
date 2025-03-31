; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 930647

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child276 - child
    bread488 - bread-portion
    content438 - content-portion
    tray334 tray39 tray131 - tray
    table472 table406 table497 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray334 kitchen)
     (at tray39 kitchen)
     (at tray131 kitchen)
     (at_kitchen_bread bread488)
     (at_kitchen_content content438)
     (not_allergic_gluten child276)
     (waiting child276 table406)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child276)
    )
  )
)
