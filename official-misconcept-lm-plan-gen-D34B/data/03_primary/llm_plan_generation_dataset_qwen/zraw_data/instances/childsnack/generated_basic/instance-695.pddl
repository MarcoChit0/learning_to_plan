; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 760872

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child476 - child
    bread15 - bread-portion
    content379 - content-portion
    tray39 tray246 tray315 tray225 - tray
    table325 table492 table334 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray39 kitchen)
     (at tray246 kitchen)
     (at tray315 kitchen)
     (at tray225 kitchen)
     (at_kitchen_bread bread15)
     (at_kitchen_content content379)
     (not_allergic_gluten child476)
     (waiting child476 table325)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child476)
    )
  )
)
