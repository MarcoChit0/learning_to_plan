; child-snack task with 1 children and 0.71 gluten factor 
; constant factor of 1.3
; random seed: 782010

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child393 - child
    bread218 - bread-portion
    content497 - content-portion
    tray235 tray264 tray443 tray68 - tray
    table353 table358 table6 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray235 kitchen)
     (at tray264 kitchen)
     (at tray443 kitchen)
     (at tray68 kitchen)
     (at_kitchen_bread bread218)
     (at_kitchen_content content497)
     (not_allergic_gluten child393)
     (waiting child393 table353)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child393)
    )
  )
)
