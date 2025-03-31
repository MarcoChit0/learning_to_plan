; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 402871

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child21 child175 - child
    bread486 bread401 - bread-portion
    content396 content2 - content-portion
    tray291 tray449 tray315 - tray
    table412 table188 table437 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray291 kitchen)
     (at tray449 kitchen)
     (at tray315 kitchen)
     (at_kitchen_bread bread486)
     (at_kitchen_bread bread401)
     (at_kitchen_content content396)
     (at_kitchen_content content2)
     (no_gluten_bread bread486)
     (no_gluten_content content396)
     (allergic_gluten child21)
     (not_allergic_gluten child175)
     (waiting child21 table437)
     (waiting child175 table188)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child21)
     (served child175)
    )
  )
)
