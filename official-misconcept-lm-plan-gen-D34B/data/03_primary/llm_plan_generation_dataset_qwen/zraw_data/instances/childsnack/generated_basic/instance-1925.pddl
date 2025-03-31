; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 224189

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child429 - child
    bread110 - bread-portion
    content232 - content-portion
    tray349 tray427 tray307 - tray
    table368 table482 table317 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray349 kitchen)
     (at tray427 kitchen)
     (at tray307 kitchen)
     (at_kitchen_bread bread110)
     (at_kitchen_content content232)
     (not_allergic_gluten child429)
     (waiting child429 table482)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child429)
    )
  )
)
