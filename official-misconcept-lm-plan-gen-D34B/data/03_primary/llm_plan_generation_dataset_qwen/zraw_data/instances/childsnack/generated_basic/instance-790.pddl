; child-snack task with 1 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 514826

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child289 - child
    bread386 - bread-portion
    content331 - content-portion
    tray437 tray223 - tray
    table121 table445 table363 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray437 kitchen)
     (at tray223 kitchen)
     (at_kitchen_bread bread386)
     (at_kitchen_content content331)
     (not_allergic_gluten child289)
     (waiting child289 table121)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child289)
    )
  )
)
