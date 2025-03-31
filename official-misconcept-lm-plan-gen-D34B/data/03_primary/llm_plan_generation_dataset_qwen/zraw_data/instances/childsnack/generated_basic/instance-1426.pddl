; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 329141

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child407 child21 - child
    bread211 bread475 - bread-portion
    content369 content494 - content-portion
    tray301 - tray
    table338 table263 table135 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray301 kitchen)
     (at_kitchen_bread bread211)
     (at_kitchen_bread bread475)
     (at_kitchen_content content369)
     (at_kitchen_content content494)
     (no_gluten_bread bread211)
     (no_gluten_content content369)
     (allergic_gluten child407)
     (not_allergic_gluten child21)
     (waiting child407 table263)
     (waiting child21 table263)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child407)
     (served child21)
    )
  )
)
