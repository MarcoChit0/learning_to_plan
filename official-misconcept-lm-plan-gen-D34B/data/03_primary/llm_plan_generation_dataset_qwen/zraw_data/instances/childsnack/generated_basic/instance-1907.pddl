; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 159925

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child439 - child
    bread296 - bread-portion
    content469 - content-portion
    tray146 tray194 tray481 - tray
    table14 table418 table211 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray146 kitchen)
     (at tray194 kitchen)
     (at tray481 kitchen)
     (at_kitchen_bread bread296)
     (at_kitchen_content content469)
     (not_allergic_gluten child439)
     (waiting child439 table418)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child439)
    )
  )
)
