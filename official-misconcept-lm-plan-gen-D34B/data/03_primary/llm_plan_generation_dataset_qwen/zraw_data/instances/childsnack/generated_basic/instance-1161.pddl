; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 158137

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child297 child41 - child
    bread204 bread460 - bread-portion
    content390 content94 - content-portion
    tray7 tray291 tray357 - tray
    table435 table67 table346 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray7 kitchen)
     (at tray291 kitchen)
     (at tray357 kitchen)
     (at_kitchen_bread bread204)
     (at_kitchen_bread bread460)
     (at_kitchen_content content390)
     (at_kitchen_content content94)
     (no_gluten_bread bread460)
     (no_gluten_content content94)
     (allergic_gluten child297)
     (not_allergic_gluten child41)
     (waiting child297 table346)
     (waiting child41 table346)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child297)
     (served child41)
    )
  )
)
