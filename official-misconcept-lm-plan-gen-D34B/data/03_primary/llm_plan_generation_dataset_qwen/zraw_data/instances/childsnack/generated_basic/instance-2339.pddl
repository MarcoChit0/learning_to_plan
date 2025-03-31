; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 217515

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child28 - child
    bread475 - bread-portion
    content315 - content-portion
    tray315 - tray
    table312 table288 table143 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray315 kitchen)
     (at_kitchen_bread bread475)
     (at_kitchen_content content315)
     (not_allergic_gluten child28)
     (waiting child28 table143)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child28)
    )
  )
)
