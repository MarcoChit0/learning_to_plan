; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 598210

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child336 - child
    bread137 - bread-portion
    content14 - content-portion
    tray313 tray238 tray353 - tray
    table399 table59 table122 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray313 kitchen)
     (at tray238 kitchen)
     (at tray353 kitchen)
     (at_kitchen_bread bread137)
     (at_kitchen_content content14)
     (not_allergic_gluten child336)
     (waiting child336 table399)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child336)
    )
  )
)
