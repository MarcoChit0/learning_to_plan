; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.3
; random seed: 724432

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child468 - child
    bread415 - bread-portion
    content197 - content-portion
    tray150 tray444 tray22 - tray
    table84 table355 table310 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray150 kitchen)
     (at tray444 kitchen)
     (at tray22 kitchen)
     (at_kitchen_bread bread415)
     (at_kitchen_content content197)
     (not_allergic_gluten child468)
     (waiting child468 table84)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child468)
    )
  )
)
