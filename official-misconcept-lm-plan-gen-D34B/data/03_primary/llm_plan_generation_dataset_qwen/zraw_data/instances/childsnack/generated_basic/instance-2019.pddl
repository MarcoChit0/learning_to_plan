; child-snack task with 2 children and 0.34 gluten factor 
; constant factor of 1.5
; random seed: 249243

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child59 child173 - child
    bread465 bread455 - bread-portion
    content61 content0 - content-portion
    tray104 tray315 tray411 - tray
    table473 table278 table256 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray104 kitchen)
     (at tray315 kitchen)
     (at tray411 kitchen)
     (at_kitchen_bread bread465)
     (at_kitchen_bread bread455)
     (at_kitchen_content content61)
     (at_kitchen_content content0)
     (not_allergic_gluten child173)
     (not_allergic_gluten child59)
     (waiting child59 table278)
     (waiting child173 table278)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child59)
     (served child173)
    )
  )
)
