; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 932360

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child73 - child
    bread457 - bread-portion
    content266 - content-portion
    tray59 tray448 tray465 - tray
    table398 table214 table106 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray59 kitchen)
     (at tray448 kitchen)
     (at tray465 kitchen)
     (at_kitchen_bread bread457)
     (at_kitchen_content content266)
     (not_allergic_gluten child73)
     (waiting child73 table398)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child73)
    )
  )
)
