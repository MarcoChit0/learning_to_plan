; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 617844

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child265 child105 - child
    bread370 bread152 - bread-portion
    content331 content61 - content-portion
    tray398 tray421 tray128 tray48 - tray
    table426 table338 table204 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray398 kitchen)
     (at tray421 kitchen)
     (at tray128 kitchen)
     (at tray48 kitchen)
     (at_kitchen_bread bread370)
     (at_kitchen_bread bread152)
     (at_kitchen_content content331)
     (at_kitchen_content content61)
     (not_allergic_gluten child105)
     (not_allergic_gluten child265)
     (waiting child265 table204)
     (waiting child105 table204)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child265)
     (served child105)
    )
  )
)
