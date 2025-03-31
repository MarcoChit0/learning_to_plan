; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 663124

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child283 child350 - child
    bread160 bread485 - bread-portion
    content217 content201 - content-portion
    tray466 tray147 - tray
    table234 table211 table344 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray466 kitchen)
     (at tray147 kitchen)
     (at_kitchen_bread bread160)
     (at_kitchen_bread bread485)
     (at_kitchen_content content217)
     (at_kitchen_content content201)
     (no_gluten_bread bread485)
     (no_gluten_content content217)
     (allergic_gluten child283)
     (not_allergic_gluten child350)
     (waiting child283 table234)
     (waiting child350 table211)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child283)
     (served child350)
    )
  )
)
