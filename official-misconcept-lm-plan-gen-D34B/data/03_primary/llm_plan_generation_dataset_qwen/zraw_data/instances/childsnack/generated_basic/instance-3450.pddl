; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 28851

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child362 - child
    bread32 - bread-portion
    content388 - content-portion
    tray333 tray158 tray160 tray450 - tray
    table140 table143 table439 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray333 kitchen)
     (at tray158 kitchen)
     (at tray160 kitchen)
     (at tray450 kitchen)
     (at_kitchen_bread bread32)
     (at_kitchen_content content388)
     (not_allergic_gluten child362)
     (waiting child362 table140)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child362)
    )
  )
)
