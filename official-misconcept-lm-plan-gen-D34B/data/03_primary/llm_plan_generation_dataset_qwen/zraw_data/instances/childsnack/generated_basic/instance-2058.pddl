; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 611480

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child420 - child
    bread28 - bread-portion
    content5 - content-portion
    tray289 tray222 tray133 tray491 - tray
    table482 table237 table70 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray289 kitchen)
     (at tray222 kitchen)
     (at tray133 kitchen)
     (at tray491 kitchen)
     (at_kitchen_bread bread28)
     (at_kitchen_content content5)
     (not_allergic_gluten child420)
     (waiting child420 table482)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child420)
    )
  )
)
