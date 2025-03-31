; child-snack task with 1 children and 0.23 gluten factor 
; constant factor of 1.7
; random seed: 481693

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child135 - child
    bread294 - bread-portion
    content398 - content-portion
    tray495 tray498 - tray
    table404 table456 table440 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray495 kitchen)
     (at tray498 kitchen)
     (at_kitchen_bread bread294)
     (at_kitchen_content content398)
     (not_allergic_gluten child135)
     (waiting child135 table404)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child135)
    )
  )
)
