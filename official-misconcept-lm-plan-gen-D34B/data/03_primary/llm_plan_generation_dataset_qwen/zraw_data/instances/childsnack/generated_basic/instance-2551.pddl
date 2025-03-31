; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 240371

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child44 - child
    bread63 - bread-portion
    content306 - content-portion
    tray252 tray395 tray479 - tray
    table467 table394 table405 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray252 kitchen)
     (at tray395 kitchen)
     (at tray479 kitchen)
     (at_kitchen_bread bread63)
     (at_kitchen_content content306)
     (not_allergic_gluten child44)
     (waiting child44 table394)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child44)
    )
  )
)
