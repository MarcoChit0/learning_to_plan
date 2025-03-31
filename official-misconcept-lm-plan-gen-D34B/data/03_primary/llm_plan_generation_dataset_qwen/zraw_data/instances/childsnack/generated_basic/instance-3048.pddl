; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 862410

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child399 child65 - child
    bread321 bread155 - bread-portion
    content352 content459 - content-portion
    tray268 tray32 tray210 - tray
    table451 table211 table402 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray268 kitchen)
     (at tray32 kitchen)
     (at tray210 kitchen)
     (at_kitchen_bread bread321)
     (at_kitchen_bread bread155)
     (at_kitchen_content content352)
     (at_kitchen_content content459)
     (no_gluten_bread bread321)
     (no_gluten_content content459)
     (allergic_gluten child65)
     (not_allergic_gluten child399)
     (waiting child399 table211)
     (waiting child65 table402)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child399)
     (served child65)
    )
  )
)
