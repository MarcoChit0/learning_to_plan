; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 867667

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child470 - child
    bread341 - bread-portion
    content288 - content-portion
    tray36 tray298 tray217 - tray
    table294 table168 table82 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray36 kitchen)
     (at tray298 kitchen)
     (at tray217 kitchen)
     (at_kitchen_bread bread341)
     (at_kitchen_content content288)
     (not_allergic_gluten child470)
     (waiting child470 table82)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child470)
    )
  )
)
