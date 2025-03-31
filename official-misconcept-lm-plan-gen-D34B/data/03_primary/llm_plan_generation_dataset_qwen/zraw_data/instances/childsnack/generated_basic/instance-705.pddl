; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 649013

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child113 - child
    bread53 - bread-portion
    content455 - content-portion
    tray408 tray44 tray21 tray199 - tray
    table381 table10 table436 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray408 kitchen)
     (at tray44 kitchen)
     (at tray21 kitchen)
     (at tray199 kitchen)
     (at_kitchen_bread bread53)
     (at_kitchen_content content455)
     (not_allergic_gluten child113)
     (waiting child113 table381)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child113)
    )
  )
)
